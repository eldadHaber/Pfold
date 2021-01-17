from supervised import log, config
from supervised.IO import dummy

if __name__ == '__main__':

   result_dir = '.'
   config.result_dir = result_dir

   logfile_loc = "{}/{}.log".format(result_dir, 'output')
   LOG = log.setup_custom_logger('runner', logfile_loc, 'debug')
   LOG.info('---------Listing all parameters-------')
   print("here")
   dummy()
   print(config.result_dir)
   print("here")
