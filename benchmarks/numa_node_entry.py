"""Optionally bind a torchrun worker to its GPU's NUMA node before Python exec.

Direct script use: python numa_node_entry.py --cpu-affinity node -- TRAINER ARGS...
Node mode requires CUDA_VISIBLE_DEVICES to list unique, complete GPU UUIDs in
the intended local CUDA order. No Torch import, CUDA calls or memory binding.
"""
import argparse
import os
from pathlib import Path
import re
import sys

_GPU_UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
_BDF = re.compile(r"([0-9a-fA-F]{4}|[0-9a-fA-F]{8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])")
_configured = False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_cpu_list(text):
    text = text.strip()
    require(bool(re.fullmatch(r"[0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*", text)),
            "Invalid NUMA CPU list")
    cpus = set()
    for part in text.split(","):
        bounds = part.split("-")
        first, last = int(bounds[0]), int(bounds[-1])
        require(first <= last, "Reversed CPU range")
        current = set(range(first, last + 1))
        require(not cpus.intersection(current), "Repeated CPU in NUMA list")
        cpus.update(current)
    return sorted(cpus)


def rank_and_uuids(environment):
    raw = environment.get("LOCAL_RANK", "")
    require(bool(re.fullmatch(r"0|[1-9][0-9]*", raw)), "Require canonical LOCAL_RANK")
    rank = int(raw)
    tokens = environment.get("CUDA_VISIBLE_DEVICES", "").split(",")
    require(tokens and all(_GPU_UUID.fullmatch(x) for x in tokens),
            "Node affinity requires complete GPU UUIDs in CUDA_VISIBLE_DEVICES")
    uuids = [x.lower() for x in tokens]
    require(len(set(uuids)) == len(uuids), "Repeated GPU UUID")
    require(rank < len(uuids), "LOCAL_RANK exceeds visible GPU list")
    world = environment.get("LOCAL_WORLD_SIZE")
    if world is not None:
        require(bool(re.fullmatch(r"[1-9][0-9]*", world)) and int(world) == len(uuids),
                "LOCAL_WORLD_SIZE must equal the explicit visible GPU list length")
    return rank, uuids


def canonical_pci_bus(value):
    match = _BDF.fullmatch(value)
    require(match is not None, "Invalid GPU PCI bus")
    domain, bus, device, function = [int(x, 16) for x in match.groups()]
    require(device <= 31, "Invalid PCI device number")
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def gpu_node(uuid, proc_root=Path("/proc/driver/nvidia/gpus"),
             pci_root=Path("/sys/bus/pci/devices"),
             node_root=Path("/sys/devices/system/node")):
    matches = []
    for information in proc_root.glob("*/information"):
        text = information.read_text()
        identifiers = re.findall(r"^GPU UUID:\s*(GPU-[0-9a-fA-F-]+)\s*$", text, re.MULTILINE)
        if uuid not in [x.lower() for x in identifiers]:
            continue
        require(len(identifiers) == 1 and _GPU_UUID.fullmatch(identifiers[0]),
                "Ambiguous GPU UUID record")
        buses = re.findall(r"^Bus Location:\s*([0-9a-fA-F:.]+)\s*$", text, re.MULTILINE)
        require(len(buses) == 1, "Missing or ambiguous GPU PCI bus")
        bdf = canonical_pci_bus(buses[0])
        require(canonical_pci_bus(information.parent.name) == bdf,
                "GPU PCI directory and bus record differ")
        matches.append(bdf)
    require(len(matches) == 1, "GPU UUID must map to exactly one visible PCI device")
    bdf = matches[0]
    pci_device = pci_root / bdf
    require(canonical_pci_bus(pci_device.resolve().name) == bdf,
            "Resolved PCI sysfs device differs")
    require((pci_device / "vendor").read_text().strip().lower() == "0x10de",
            "GPU PCI vendor is not NVIDIA")
    node_text = (pci_device / "numa_node").read_text().strip()
    require(bool(re.fullmatch(r"0|[1-9][0-9]*", node_text)), "GPU NUMA node is unknown")
    cpus = parse_cpu_list((node_root / ("node" + node_text) / "cpulist").read_text())
    require(cpus, "GPU NUMA node has no CPUs")
    return bdf, int(node_text), cpus


def require_before_torch():
    require(sys.modules.get("torch") is None, "CPU binding must precede Torch import")
    tids = sorted(int(p.name) for p in Path("/proc/self/task").iterdir())
    require(tids == [os.getpid()], "CPU binding requires a single main thread")


def configure_affinity(mode, environment=None):
    global _configured
    require(not _configured, "CPU affinity entry can only be configured once")
    require(mode in ("off", "node"), "CPU affinity must be off or node")
    require_before_torch()
    environment = os.environ if environment is None else environment
    original = sorted(os.sched_getaffinity(0))
    require(original, "Original allowed CPU set is empty")
    result = {"mode": mode, "pid": os.getpid(), "original_cpus": original,
              "selected_cpus": original, "setter_calls": 0}
    if mode == "node":
        rank, uuids = rank_and_uuids(environment)
        bdf, node, cpus = gpu_node(uuids[rank])
        require(set(cpus).issubset(original),
                "The full GPU NUMA node is outside the original allowed CPU set")
        require_before_torch()
        require(sorted(os.sched_getaffinity(0)) == original,
                "Allowed CPU set changed before binding")
        os.sched_setaffinity(0, cpus)
        require(sorted(os.sched_getaffinity(0)) == cpus, "CPU affinity did not match the requested node")
        require_before_torch()
        result.update(local_rank=rank, gpu_uuid=uuids[rank], PCI_BDF=bdf,
                      numa_node=node, selected_cpus=cpus, setter_calls=1)
    _configured = True
    return result


def interpreter_flags():
    """Preserve the actual flags of direct-script Python/torchrun invocation."""
    require(Path(sys.argv[0]).resolve() == Path(__file__).resolve(),
            "Invoke this entry as a Python script, not through -m/-c or an imported main")
    original = getattr(sys, "orig_argv", None)
    require(isinstance(original, list) and len(original) >= len(sys.argv) + 1
            and original[-len(sys.argv):] == sys.argv,
            "Cannot identify the original Python flags without guessing")
    return original[1:len(original) - len(sys.argv)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-affinity", choices=("off", "node"), default="off")
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- TRAINER ARGS")
    args = parser.parse_args()
    command = args.command
    require(command[:1] == ["--"] and len(command) > 1,
            "Pass the original trainer script and arguments after --")
    command = command[1:]
    require(command[0] and not command[0].startswith("-") and Path(command[0]).is_file(),
            "Original trainer must be an existing Python script")
    flags = interpreter_flags()
    binding = configure_affinity(args.cpu_affinity)
    os.execv(sys.executable, [sys.executable, *flags, *command])


if __name__ == "__main__":
    main()
