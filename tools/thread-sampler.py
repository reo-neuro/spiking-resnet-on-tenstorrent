# Sample the threads of a running process: every <interval> s, for each thread that used CPU since the last sample,
# print the epoch, tid, name, the CPU it last ran on, its CPU time in the interval (ms) and that CPU's current
# frequency: scaling_cur_freq (with acpi-cpufreq this is the requested P-state, 2101 = "turbo") and the
# /proc/cpuinfo "cpu MHz" (APERF/MPERF based, i.e. the actual average clock incl. turbo). Read-only (/proc, /sys).
# Run: python3 tools/thread-sampler.py <pid> [interval=0.1] [duration=120] > out.csv
import glob
import os
import sys
import time

pid = int(sys.argv[1])
iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
dur = float(sys.argv[3]) if len(sys.argv) > 3 else 120
tick_ms = 1000 / os.sysconf("SC_CLK_TCK")


def threads():
    out = {}
    for d in glob.glob(f"/proc/{pid}/task/*"):
        try:
            st = open(d + "/stat").read()
        except OSError:
            continue
        name = st[st.index("(") + 1 : st.rindex(")")]
        f = st[st.rindex(")") + 2 :].split()  # fields from 3 (state) on
        out[int(os.path.basename(d))] = (name, int(f[11]) + int(f[12]), int(f[36]))  # utime+stime, processor
    return out


def freq(cpu):
    try:
        return int(open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read()) // 1000
    except OSError:
        return -1


def actual_mhz():
    out, cpu = {}, None
    for line in open("/proc/cpuinfo"):
        if line.startswith("processor"):
            cpu = int(line.split(":")[1])
        elif line.startswith("cpu MHz"):
            out[cpu] = float(line.split(":")[1])
    return out


print("epoch,tid,name,cpu,cpu_ms,freq_mhz,actual_mhz", flush=True)
prev = threads()
end = time.time() + dur
while time.time() < end and os.path.exists(f"/proc/{pid}"):
    time.sleep(iv)
    now, cur, mhz = time.time(), threads(), actual_mhz()
    for tid, (name, t, cpu) in cur.items():
        d = t - prev.get(tid, (name, t, cpu))[1]
        if d > 0:
            print(f"{now:.3f},{tid},{name},{cpu},{d * tick_ms:.0f},{freq(cpu)},{mhz.get(cpu, -1):.0f}", flush=True)
    prev = cur
