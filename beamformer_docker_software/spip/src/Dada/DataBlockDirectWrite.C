/***************************************************************************
 *
 *   Copyright (C) 2015 Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "spip/DataBlockDirectWrite.h"

#include <stdexcept>

using namespace std;

spip::DataBlockDirectWrite::DataBlockDirectWrite (const char * key_string) : DataBlockDirect(key_string)
{
  db->lock_write();
  
  is_block_open = false;
  curr_block_id = 0;
  curr_block_ptr = 0;
}

spip::DataBlockDirectWrite::~DataBlockDirectWrite ()
{
  db->unlock_write();
}

void * spip::DataBlockDirectWrite::open_block ()
{
  if (is_block_open)
    throw runtime_error ("block already open");

  curr_block_ptr = (void *) ipcio_open_block_write (db->get_data_block(), &curr_block_id);

  is_block_open = true;

  return curr_block_ptr;
}

void spip::DataBlockDirectWrite::close_block ()
{
  if (!is_block_open)
    throw runtime_error ("block was not open");

  is_block_open = false;
  curr_block_ptr = 0;  
  curr_block_id = 0;
}

void spip::DataBlockDirectWrite::add_bytes_written (uint64_t bytes)
{
  curr_block_bytes += bytes;
}

