import sys
from composer.utils import dist, get_device
import os
import logging

if __name__ == '__main__':
  logging.basicConfig(
    # Example of format string
    # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
    format=
    f'%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s'
  )

  logging.getLogger(__name__).setLevel(
      'DEBUG') 
  
  log = logging.getLogger(__name__)


  dir_path = os.path.abspath(sys.argv[1])
  dist.initialize_dist(get_device(None), timeout=600)

  file_list = [[
    os.path.join(dir_path, file_name) for file_name in sorted(os.listdir(dir_path)) if file_name.endswith('.distcp')
  ]]
  dist.broadcast_object_list(file_list, src=0)
  file_list = file_list[0]
  log.debug(f'List of files to broadcast: {file_list}')

  for file_path in file_list:
    if dist.get_global_rank() == 0:
      with open(file_path, 'rb') as f:
          file_object = [{'content': f.read()}]
    else:
      file_object = [None]

    log.debug(f'Starting broadcast: {file_path}')
    dist.broadcast_object_list(
      object_list=file_object,
      src=0
    )
    log.debug(f'Finished broadcast: {file_path}')
    received_file_object = file_object[0]
    assert received_file_object is not None
    if dist.get_global_rank() != 0 and dist.get_local_rank() == 0:
        log.debug(f'Writing file: {file_path}')
        with open(file_path, 'wb') as f:
          f.write(received_file_object['content'])
  dist.barrier()