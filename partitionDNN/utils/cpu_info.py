import platform
import subprocess
import fileinput

import psutil
import GPUtil


def get_mac_cpu_speed():
    commond = 'system_profiler SPHardwareDataType | grep "Processor Speed" | cut -d ":" -f2'
    proc = subprocess.Popen([commond], shell=True, stdout=subprocess.PIPE)
    output = proc.communicate()[0]
    output = output.decode()   # bytes 转str
    speed = output.lstrip().rstrip('\n')
    return speed


def get_linux_cpu_speed():
    for line in fileinput.input('/proc/cpuinfo'):
        if 'MHz' in line:
            value = line.split(':')[1].strip()
            value = float(value)
            speed = round(value / 1024, 1)
            return "{speed} GHz".format(speed=speed)


def get_windows_cpu_speed():
    import winreg
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
    speed, type = winreg.QueryValueEx(key, "~MHz")
    speed = round(float(speed)/1024, 1)
    return speed
    # return "{speed} GHz".format(speed=speed)


def get_cpu_speed():
    osname = platform.system()  # 获取操作系统的名称
    speed = ''
    if osname == "Darwin":
        speed = get_mac_cpu_speed()
    if osname == "Linux":
        speed = get_linux_cpu_speed()
    if osname in ["Windows", "Win32"]:
        speed = get_windows_cpu_speed()
    return speed

# cpu信息
def get_cpu_used():
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_info = "CPU使用率：%i%%" % cpu_percent
    # print(cpu_info)
    return cpu_percent

# 内存信息
def get_memory_info():
    virtual_memory = psutil.virtual_memory()
    used_memory = virtual_memory.used/1024/1024/1024
    free_memory = virtual_memory.free/1024/1024/1024
    memory_percent = virtual_memory.percent
    memory_info = "内存使用：%0.2fG，使用率%0.1f%%，剩余内存：%0.2fG" % (used_memory, memory_percent, free_memory)
    # print(memory_info)
    return free_memory



def gpu_util_timer():
    Gpus = GPUtil.getGPUs()
    for gpu in Gpus:
        return gpu.memoryFree

def get_platform_capability():
    """获取当前设备的计算能力"""
    cpu_all = get_cpu_speed() # CPU总的Hz 2.1GHz
    cpu_used = get_cpu_used() # CPU已经使用了白分比
    cpu_free = cpu_all * (100 - cpu_used)  # 剩余CPU多少GHz
    memory_free = get_memory_info()  # 剩余内存G，4GB
    gpu_free = gpu_util_timer() / 1024  # 剩余GPU多少G，1GB
    print(gpu_free)
    capability = float(cpu_free * 0.3 + memory_free * 0.3 + gpu_free * 0.4) # 加权求和
    return capability

if __name__ == '__main__':
    speed = get_cpu_speed()
    print(speed)
    cpu_used = get_cpu_used()
    print(cpu_used)
    free_memory = get_memory_info()
    print("free",free_memory)
    print(gpu_util_timer())
    get_platform_capability()

