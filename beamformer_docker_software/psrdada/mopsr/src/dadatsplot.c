/*
 * read a file from disk and create the associated images
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <errno.h>
#include <float.h>
//#include <complex.h>
#include <math.h>
#include <cpgplot.h>

#include "dada_def.h"
#include "mopsr_def.h"
#include "mopsr_util.h"
#include "mopsr_udp.h"

#include "string_array.h"
#include "ascii_header.h"
#include "daemon.h"

#define CHECK_ALIGN(x) assert ( ( ((uintptr_t)x) & 15 ) == 0 )

typedef struct {

  // pgplot device
  char * device;

  // number of antennae
  unsigned int npol;

  // number of input channels
  unsigned int nchan_in;

  // number of FFT points to perform on input channels
  unsigned int npt;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // which channel to plot 
  int channel;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  float * x_ts;

  float ** y_ts;

  float ymin;

  float ymax;

  float xmin;

  float xmax;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void append_samples (udpplot_t * ctx, void * buffer,  unsigned npt, uint64_t nbytes);
void plot_ts (udpplot_t * ctx);

void usage ();


void usage()
{
  fprintf (stdout,
     "dadatsplot [options] dadafile\n"
     " -c channel  set the channel to plot\n"
     " -n npt      number of points in each timeseries [default 1024]\n"
     " -D device   pgplot device name\n"
     " -v          be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int npt = 1024;

  unsigned int plot_log = 0;

  float xmin = 0;
  float xmax = 0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  unsigned channel = 263;

  while ((arg=getopt(argc,argv,"c:D:n:vz")) != -1)
  {
    switch (arg)
    {
      case 'c':
        channel = atoi(optarg);
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'n':
        npt = atoi(optarg);
        break;

      case 'v':
        verbose++;
        break;

        usage ();
        return 0;
    } 
  }

  // check and parse the command line arguments
  if (argc-optind != 1)
  {
    fprintf(stderr, "ERROR: 1 command line arguments are required\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  char filename[256];
  char png_file[256];
  strcpy(filename, argv[optind]);

  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t filesize = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %ld bytes\n", filename, filesize);

  int flags = O_RDONLY;
  int perms = S_IRUSR | S_IRGRP;
  int fd = open (filename, flags, perms);
  if (fd < 0)
  {
    fprintf(stderr, "failed to open dada file[%s]: %s\n", filename, strerror(errno));
    exit (EXIT_FAILURE);
  }

  // get the size of the ascii header in this file
  size_t hdr_size = ascii_header_get_size_fd (fd);
  char * header = (char *) malloc (hdr_size + 1);

  if (verbose)
    fprintf (stderr, "reading header, %ld bytes\n", hdr_size);
  size_t bytes_read = read (fd, header, hdr_size);
  size_t data_size = filesize - hdr_size;

  udpplot_t udpplot;
  udpplot.verbose = verbose;
  udpplot.ndim = 2;
  udpplot.npol = 2;
  udpplot.channel = channel;

  if (ascii_header_get(header, "NCHAN", "%d", &(udpplot.nchan_in)) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }
  udpplot.npt = npt;

  fprintf (stderr, "nchan=%d npt=%d\n", udpplot.nchan_in, udpplot.npt);

  if (header)
    free(header);
  header = 0;

  udpplot.xmin = xmin;
  udpplot.xmax = xmax;
  udpplot.ymin = ymin;
  udpplot.ymax = ymax;

  if (verbose)
    fprintf(stderr, "mopsr_dadatsplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_dadatsplot: error opening plot device\n");
    exit(1);
  }

  //cpgask(1);

  udpplot_t * ctx = &udpplot;

  // allocate require resources
  if (udpplot_init (ctx) < 0)
  {
    fprintf (stderr, "ERROR: Could not alloc memory\n");
    exit(1);
  }

  // cloear packets ready for capture
  udpplot_reset (ctx);
  size_t read_size = (size_t) npt * ctx->ndim * ctx->npol * ctx->nchan_in;
  fprintf (stderr, "npt=%u ndim=%u npol=%u nchan=%u\n", npt, ctx->ndim, ctx->npol, ctx->nchan_in);
  size_t total_bytes_read = 0;

  if (verbose)
    fprintf (stderr, "mopsr_dadatsplot: allocating %ld bytes\n", read_size);
  void * raw = malloc (read_size);

  while (total_bytes_read < data_size)
  {
    bytes_read = read (fd, raw, read_size);
    if (bytes_read == read_size)
    {
      append_samples (ctx, raw, ctx->npt, read_size);
      plot_ts(ctx);
      udpplot_reset (ctx);
      sleep (1);
    }
    total_bytes_read += bytes_read;
  }

  udpplot_destroy (ctx);
  cpgclos();
  close(fd);

  if (raw)
    free (raw);

  return EXIT_SUCCESS;
}

int udpplot_reset (udpplot_t * ctx)
{
}

int udpplot_destroy (udpplot_t * ctx)
{
  free (ctx->y_ts[0]);
  free (ctx->y_ts[1]);
  free (ctx->y_ts[2]);
  free (ctx->y_ts[3]);
  free (ctx->y_ts);
  free (ctx->x_ts);
}

int udpplot_init (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    fprintf(stderr, "mopsr_udpdb_init_receiver()\n");
  ctx->x_ts = (float *) malloc (sizeof(float) * ctx->npt);
  ctx->y_ts = (float **) malloc(sizeof(float *) * ctx->ndim * ctx->npol);
  ctx->y_ts[0] = (float *) malloc(sizeof(float) * ctx->npt);
  ctx->y_ts[1] = (float *) malloc(sizeof(float) * ctx->npt);
  ctx->y_ts[2] = (float *) malloc(sizeof(float) * ctx->npt);
  ctx->y_ts[3] = (float *) malloc(sizeof(float) * ctx->npt);
  unsigned ipt;
  for (ipt=0; ipt< ctx->npt; ipt++)
  {
    ctx->x_ts[ipt] = ipt;
  }

  return 0;
}

// copy data from packet in the fft input buffer
void append_samples (udpplot_t * ctx, void * buffer, unsigned npt, size_t nbytes)
{
  unsigned ichan, ipol, iheap, idat;

  unsigned nheap = npt / 256;
  int8_t * in = (int8_t *) buffer ;

  for (iheap=0; iheap<nheap; iheap++)
  {
    for (ipol=0; ipol < ctx->npol; ipol++)
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        for (idat=0; idat<256; idat++)
        {
          if (ichan == ctx->channel)
          {
            ctx->y_ts[ctx->ndim*ipol+0][iheap*256+idat] = (float) in[0];
            ctx->y_ts[ctx->ndim*ipol+1][iheap*256+idat] = (float) in[1];
          }

          in += 2;
        }
      }
    }
  }
}

void plot_ts (udpplot_t * ctx)
{
  if (ctx->verbose)
    fprintf(stderr, "plot_ts()\n");

  int npt = ctx->npt;
  float xmin = 0;
  float xmax = (float) npt;
  float ymin = -140;
  float ymax = 140;
  char lines = 1;

  cpgeras();

  cpgbbuf();
  cpgsci(1);

  cpgswin(xmin, xmax, ymin, ymax);
  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "8-bit Voltage", "");

  cpgmtxt("T", -2.2, 0.05, 0.0, "Pol 0");
  if (lines)
  {
    cpgsci(2);
    cpgline(npt, ctx->x_ts, ctx->y_ts[0]);
    cpgsci(3);
    cpgline(npt, ctx->x_ts, ctx->y_ts[1]);
  }
  else
  {
    cpgsci(2);
    cpgslw(3);
    cpgpt(npt, ctx->x_ts, ctx->y_ts[0], -1);
    cpgsci(3);
    cpgpt(npt, ctx->x_ts, ctx->y_ts[1], -1);
    cpgslw(1);
  }
  cpgsci(1);

  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("Sample", "8-bit Voltage", "");

  cpgmtxt("T", -2.2, 0.05, 0.0, "Pol 1");

  if (lines)
  {
    cpgsci(2);
    cpgline(npt, ctx->x_ts, ctx->y_ts[2]);
    cpgsci(3);
    cpgline(npt, ctx->x_ts, ctx->y_ts[3]);
  }
  else
  {
    cpgslw(3);
    cpgsci(2);
    cpgpt(npt, ctx->x_ts, ctx->y_ts[2], -1);
    cpgsci(3);
    cpgpt(npt, ctx->x_ts, ctx->y_ts[3], -1);
    cpgslw(1);
  }
  cpgsci(1);

  cpgebuf();
}

