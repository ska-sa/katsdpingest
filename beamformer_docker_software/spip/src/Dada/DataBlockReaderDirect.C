/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DataBlockReaderDirect.h"

#include <sstream>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

using namespace std;

spip::DataBlockReaderDirect::DataBlockReaderDirect (const char * key_string)
{
  key_t key;

  stringstream ss;
  ss << std::hex << key_string;
  ss >> key;

  cerr << "key=" << key;

  // parse key_string into a dada key
  //if (sscanf (key_string, "%x", &key) != 1) 
  //{
  //  cerr << "spip::DataBlockReaderDirect::DataBlockReaderDirect could not parse " << key_string 
  //       << " as PSRDada key";
  //  throw runtime_error ("Bad PSRDada key");
  //}

  // keys for the header + data unit
  data_block_key = key;
  header_block_key = key + 1;

  ipcbuf_t ipcbuf_init = IPCBUF_INIT;
  ipcio_t ipcio_init = IPCIO_INIT;

  header_block = (ipcbuf_t *) malloc (sizeof(ipcbuf_t));
  *header_block = ipcbuf_init;

  data_block = (ipcio_t *) malloc (sizeof(ipcio_t));
  *data_block = ipcio_init;
  
  connected = false;
  locked_read = false;
  locked_write = false;
  locked_view = false;

}

spip::DataBlockReaderDirect::~DataBlockReaderDirect ()
{
  if (header_block)  
    free (header_block);
  if (data_block)
    free(data_block);
}

void spip::DataBlockReaderDirect::connect ()
{
  if (connected)
    throw runtime_error ("already connected to data block");

  if (ipcbuf_connect (header_block, header_block_key) < 0)
    throw runtime_error ("failed to connect to header block");
  if (ipcio_connect (data_block, data_block_key) < 0)
    throw runtime_error ("failed to connect to data block");
  
  connected = true;
}

void spip::DataBlockReaderDirect::disconnect ()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (ipcio_disconnect (data_block) < 0) 
    throw runtime_error ("failed to disconnect from data block");
  if (ipcbuf_disconnect (header_block) < 0)
    throw runtime_error ("failed to disconnect from header block");

  connected = false;
}

void spip::DataBlockReaderDirect::lock_read ()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (locked_read || locked_write || locked_view)
    throw runtime_error ("not unlocked");

  if (ipcbuf_lock_read (header_block) < 0) 
    throw runtime_error ("could not lock header block for reading");

  if (ipcio_open (data_block, 'R') < 0) 
   throw runtime_error ("could not lock header block for writing");

  locked_read = true;
}

void spip::DataBlockReaderDirect::unlock_read()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (!locked_read || locked_write || locked_view)
    throw runtime_error ("not locked for reading on data block");

  if (ipcio_close (data_block) < 0)
    throw runtime_error ("could not unlock data block from reading");

  if (header)
    free (header);
  header = 0;

  if (ipcbuf_is_reader (header_block))
   ipcbuf_mark_cleared (header_block);

  if (ipcbuf_unlock_read (header_block) < 0)
   throw runtime_error ("could not unlock header block from reading");

  locked_read = false;
}

void spip::DataBlockReaderDirect::lock_write ()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (locked_write || locked_read || locked_view)
    throw runtime_error ("not unlocked");

  if (ipcbuf_lock_write (header_block) < 0)
    throw runtime_error ("could not lock header block for writing");

  if (ipcio_open (data_block, 'W') < 0)
    throw runtime_error ("could not lock data block for writing");

  locked_write = true;
}

void spip::DataBlockReaderDirect::unlock_write ()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (!locked_write || locked_read || locked_view)
    throw runtime_error ("not locked for writing on data block");

  if (ipcio_is_open (data_block))
    if (ipcio_close (data_block) < 0)
      throw runtime_error ("could not unlock data block from writing");

  if (ipcbuf_unlock_write (header_block) < 0)
    throw runtime_error ("could not unlock header block from writing");

  locked_write = true;
}

void spip::DataBlockReaderDirect::lock_view ()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (locked_view || locked_write || locked_read)
    throw runtime_error ("not unlocked");

  if (ipcio_open (data_block, 'r') < 0)
    throw runtime_error ("could not open data block for viewing");

  locked_view = true;
}

void spip::DataBlockReaderDirect::unlock_view ()
{
  if (!connected)
    throw runtime_error ("not connected to data block");

  if (!locked_view || locked_write || locked_read)
    throw runtime_error ("data block not locked for viewing");

  if (ipcio_close (data_block) < 0)
    throw runtime_error ("could not close data block from viewing");

  locked_view = false;
}
