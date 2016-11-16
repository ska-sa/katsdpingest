#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

import os, threading, sys, time, socket, select, signal, traceback
import spip
import spip_smrb

SCRIPT = "spip_recv"
DL     = 2

#################################################################
# main
def main (argv):

  # this should come from command line argument
  stream_id = argv[1]

  # read configuration file
  cfg = spip.getConfig()

  control_thread = []

  log_file  = cfg["SERVER_LOG_DIR"] + "/" + SCRIPT + ".log"
  pid_file  = cfg["SERVER_CONTROL_DIR"] + "/" + SCRIPT + ".pid"
  quit_file = cfg["SERVER_CONTROL_DIR"] + "/"  + SCRIPT + ".quit"

  if os.path.exists(quit_file):
    sys.stderr.write("quit file existed at launch: " + quit_file)
    sys.exit(1)

  # become a daemon
  # spip.daemonize(pid_file, log_file)

  try:

    spip.logMsg(1, DL, "STARTING SCRIPT")

    quit_event = threading.Event()

    def signal_handler(signal, frame):
      sys.stderr.write("SIGNAL: quit_event.set()\n")
      quit_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # start a control thread to handle quit requests
    control_thread = spip.controlThread(quit_file, pid_file, quit_event, DL)
    control_thread.start()

    log_host = cfg["SERVER_HOST"]
    log_port = int(cfg["SERVER_LOG_PORT"])

    log_pipe = spip.openSocket (DL, log_host, log_port, 3)

    db_id = cfg["PROCESSING_DATA_BLOCK"]
    db_prefix = cfg["DATA_BLOCK_PREFIX"]
    num_stream = cfg["NUM_STREAM"]
    db_key = spip_smrb.getDBKey (db_prefix, stream_id, num_stream, db_id)
    spip.logMsg(0, DL, "db_key="+db_key)

    # wait up to 10s for the SMRB to be created
    smrb_wait = 10
    cmd = "dada_dbmetric -k " + db_key
    rval = 1
    while rval and smrb_wait > 0:
      rval, lines = spip.system(cmd, 3 <= DL)
      if rval:
        time.sleep(1)
      smrb_wait -= 1

    if rval:
      spip.logMsg(-2, DL, "spip_smrb["+str(stream_id)+"] no valid SMRB with " +
                  "key=" + db_key)
      quit_event.set()

    else:
   
      ctrl_port = str(int(cfg["STREAM_CTRL_PORT"]) + int(stream_id))
      log_port  = str(int(cfg["STREAM_LOG_PORT"])  + int(stream_id))
      stream_core = cfg["STREAM_CORE_" + str(stream_id)]  
      (stream_ip, stream_port) =  cfg["STREAM_UDP_" + str(stream_id)].split(":")

      cmd = cfg["STREAM_BINARY"] + " -b " + stream_core + " -c " + ctrl_port \
            + " -k " + db_key + " -i " + stream_ip   + " -l " + log_port \
            + " -p " + stream_port

      # this should be a persistent / blocking command 
      rval = spip.system_piped (cmd, log_pipe, 2 <= DL)
      if rval:
        quit_event.set()

  except:
    spip.logMsg(-2, DL, "main: exception caught: " + str(sys.exc_info()[0]))
    print '-'*60
    traceback.print_exc(file=sys.stdout)
    print '-'*60
    quit_event.set()

  # join threads
  spip.logMsg(2, DL, "main: joining control thread")
  if (control_thread):
    control_thread.join()

  spip.logMsg(1, DL, "STOPPING SCRIPT")

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print "ERROR: 1 command line argument expected"
    sys.exit(1)

  main (sys.argv)
  sys.exit(0)
