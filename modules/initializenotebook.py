import tensorflow as tf
import subprocess
from psutil import virtual_memory


def check_GPU_availability(show_details=False):
    # Show GPU name
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        raise SystemError("GPU device not found")
    print("Found GPU at: {}".format(device_name))
    if show_details:
        # Show GPU type
        gpu_info = subprocess.run(["!nvidia-smi", ""], stdout=subprocess.PIPE)
        gpu_info = "\n".join(gpu_info)
        if gpu_info.find("failed") >= 0:
            print(
                'Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, '
            )
            print("and then re-execute this cell.")
        else:
            print(gpu_info)


def check_RAM_availability():
    ram_gb = virtual_memory().total / 1e9
    print("Your runtime has {:.1f} gigabytes of available RAM\n".format(ram_gb))

    if ram_gb < 20:
        print(
            'To enable a high-RAM runtime, select the Runtime > "Change runtime type"'
        )
        print("menu, and then select High-RAM in the Runtime shape dropdown. Then, ")
        print("re-execute this cell.")
    else:
        print("You are using a high-RAM runtime!")


def check_hardware(show_details=False):
    check_GPU_availability(show_details)
    check_RAM_availability()
    strategy = tf.distribute.MirroredStrategy()
    return strategy
