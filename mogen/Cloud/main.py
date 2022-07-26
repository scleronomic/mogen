import time

from wzk.subprocess2 import popen_list
from wzk.gcp.gcloud2 import *
from wzk.gcp import startup


def create_instances_and_start(mode='genpath', n=10, n0=0,
                               disk_size=20,
                               sleep=600):
    machine = 'c2-standard-60'
    snapshot = 'tenh-setup-cpu'
    snapshot_size = 500
    startup_script = startup.make_startup_file(user=GCP_USER,
                                               bash_file=f"/home/{GCP_USER}/src/mogen/mogen/Cloud/Startup/{mode}.sh")

    instance_list = [f"{GCP_USER_SHORT}-{mode}-{n0+i}" for i in range(n)]
    disk_list = [f"{GCP_USER_SHORT}-{mode}-disk-{n0+i}" for i in range(n)]

    cmd_disks = []
    cmd_instances = []
    cmd_attach_disks = []
    for i in range(n):
        disk = dict(name=disk_list[i], size=disk_size, labels=GCP_USER_LABEL)
        disk_boot = dict(name=instance_list[i], snapshot=snapshot, size=snapshot_size, autodelete='yes', boot='yes')
        instance = dict(name=instance_list[i],
                        machine=machine, disks_new=disk_boot, disks_old=None, disks_local=None,
                        startup_script=startup_script, labels=GCP_USER_LABEL)

        cmd_disks.append(create_disk_cmd(disk=disk))
        cmd_instances.append(create_instance_cmd(config=instance))
        cmd_attach_disks.append(attach_disk_cmd(instance=instance, disk=disk))

    popen_list(cmd_list=cmd_disks)

    for a, b in zip(cmd_instances, cmd_attach_disks):
        subprocess.call(a, shell=True)
        subprocess.call(b, shell=True)
        if len(cmd_attach_disks) > 1:
            time.sleep(sleep)


# TODO add script to mount disk

def mogen_create_instance_local(i):
    name = f"tenh-sql-{i}"
    machine = 'c2-standard-60'
    snapshot = 'tenh-setup-cpu'
    snapshot_size = 30
    startup_script = startup.make_startup_file(user=GCP_USER,
                                               bash_file=f"/home/{GCP_USER}/src/mogen/mogen/Cloud/Startup/basic.sh")

    disk_boot = dict(name=name, snapshot=snapshot, size=snapshot_size, autodelete='yes', boot='yes')
    disk_local = dict(interface='SCSI', n=8)
    instance = dict(name=name,
                    machine=machine, disks_new=disk_boot, disks_old=None, disks_local=disk_local,
                    startup_script=startup_script, labels=GCP_USER_LABEL)
    cmd = create_instance_cmd(instance)
    subprocess.call(cmd, shell=True)


def mogen_upload2bucket(robot_id, n, n0=0, prefix='', disk_name=''):
    prefix = f'{prefix}_' if prefix else ''
    disk = f"tenh-{disk_name}-disk"
    file = f'/home/{GCP_USER}/sdb/{prefix}{robot_id}.db'
    bucket = f'gs://tenh_jo/{prefix}{robot_id}'
    upload2bucket(disk=disk, file=file, bucket=bucket, n=n, n0=n0)


if __name__ == '__main__':
    fire.Fire({
        'create_start': create_instances_and_start,
        'create_local': mogen_create_instance_local,
        'upload': mogen_upload2bucket,
        'connect2': connect2,
    })
