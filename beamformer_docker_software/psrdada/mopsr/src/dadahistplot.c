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

  unsigned int npt;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  unsigned int nbin;

  // which channel to plot 
  int channel;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  unsigned ** histograms;

  uint64_t num_integrated;

  uint64_t to_integrate;


} udphist_t;

int udphist_init (udphist_t * ctx);
int udphist_prepare (udphist_t * ctx);
int udphist_destroy (udphist_t * ctx);

void append_samples (udphist_t * ctx, void * buffer, unsigned npt, uint64_t nbytes);
void plot_data (udphist_t * ctx);

void usage ();


void usage()
{
  fprintf (stdout,
     "dadahistplot [options] dadafile\n"
     " -c channel  show histogram for channel\n"
     " -D device   pgplot device name\n"
     " -t num      number of samples to histogram into each plot\n"
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

  unsigned int to_integrate = 8;

  int channel = -1;

  while ((arg=getopt(argc,argv,"c:D:n:t:vz")) != -1)
  {
    switch (arg)
    {
      case 'c':
        channel = atoi (optarg); 
        break;

      case 'D':
        device = strdup(optarg);
        break;

      case 'n':
        npt = atoi(optarg);
        break;

      case 't':
        to_integrate = atoi (optarg);
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
  strcpy(filename, argv[optind]);

  struct stat buf;
  if (stat (filename, &buf) < 0)
  {
    fprintf (stderr, "ERROR: failed to stat dada file [%s]: %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }

  size_t filesize = buf.st_size;
  if (verbose)
    fprintf (stderr, "filesize for %s is %d bytes\n", filename, filesize);

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
  if (bytes_read != hdr_size)
  {
    fprintf (stderr, "failed to read %ld bytes of header\n", hdr_size);
    exit (EXIT_FAILURE);
  }

  size_t data_size = filesize - hdr_size;


  udphist_t udphist;
  udphist.verbose = verbose;
  udphist.channel = channel;
  udphist.nbin = 256;

  if (ascii_header_get(header, "NDIM", "%u", &(udphist.ndim)) != 1)
  {
    fprintf (stderr, "could not extract NDIM from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "NPOL", "%u", &(udphist.npol)) != 1)
  {
    fprintf (stderr, "could not extract NPOLfrom header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "NCHAN", "%d", &(udphist.nchan_in)) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }
  udphist.npt = npt;

  if (header)
    free(header);
  header = 0;

  udphist.num_integrated = 0;
  udphist.to_integrate = to_integrate;

  if (verbose)
    fprintf(stderr, "mopsr_dadafftplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_dadafftplot: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  udphist_t * ctx = &udphist;

  // allocate require resources
  if (udphist_init (ctx) < 0)
  {
    fprintf (stderr, "ERROR: Could not alloc memory\n");
    exit(1);
  }

  // cloear packets ready for capture
  udphist_reset (ctx);

  size_t read_size = (size_t) npt * ctx->ndim * ctx->npol * ctx->nchan_in;
  size_t total_bytes_read = 0;
  void * raw = malloc (read_size);

  while (total_bytes_read < data_size)
  {
    bytes_read = read (fd, raw, read_size);
    if (bytes_read == read_size)
    {
      append_samples (ctx, raw, ctx->npt, read_size);
      unsigned i;
      for (i=126; i<=130; i++)
      {
        ctx->histograms[0][i] = 0;
        ctx->histograms[1][i] = 0;
        ctx->histograms[2][i] = 0;
        ctx->histograms[3][i] = 0;
      }
      plot_data (ctx);
      udphist_reset (ctx);
      sleep(1);
    }
  }

  udphist_destroy (ctx);
  cpgclos();
  close(fd);

  if (raw)
    free (raw);

  return EXIT_SUCCESS;
}

int udphist_reset (udphist_t * ctx)
{
  unsigned ipol, idim, ibin;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (idim=0; idim < ctx->ndim; idim++)
    {
      for (ibin=0; ibin < ctx->nbin; ibin++)
      {
        ctx->histograms[2*ipol+idim][ibin] = 0;
      }
    }
  }
  ctx->num_integrated = 0;
}

int udphist_destroy (udphist_t * ctx)
{

  unsigned int ipol;
  unsigned int idim;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    for (idim=0; idim < ctx->ndim; idim++)
    {
      free (ctx->histograms[2*ipol+idim]);
    }
  }
  free (ctx->histograms);
}

int udphist_init (udphist_t * ctx)
{
  ctx->histograms = (unsigned **) malloc (sizeof(unsigned *) * ctx->npol * ctx->ndim);
  unsigned ipol, idim;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    for (idim=0; idim < ctx->ndim; idim++)
    {
      ctx->histograms[2*ipol+idim] = (unsigned *) malloc (sizeof(unsigned) * ctx->nbin);
    }
  }
  return 0;
}

// copy data from packet in the fft input buffer
void append_samples (udphist_t * ctx, void * buffer, unsigned npt, uint64_t nbytes)
{
  unsigned ichan, ipol, ipt, iheap, idat;

  //unsigned nheap = nbytes / (ctx->nchan_in * ctx->npol * ctx->ndim * 256);
  unsigned nheap = npt / 256;

  int8_t * in = (int8_t *) buffer;
  int re, im;
  for (iheap=0; iheap<nheap; iheap++)
  {
    for (ipol=0; ipol < ctx->npol; ipol++)
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        for (idat=0; idat<256; idat++)
        {
          if (ichan == ctx->channel || ctx->channel == -1)
          {
            re = ((int) in[0]) + 128;
            im = ((int) in[1]) + 128;
            //fprintf (stderr, "[%u][%u][%u][%u] %d->%d %d->%d\n", iheap, ipol, ichan, idat, in[0], re, in[1], im);
            ctx->histograms[2*ipol+0][re]++;
            ctx->histograms[2*ipol+1][im]++;
          }
          in += 2;
        }
      }
    }
    ipt += 256;
  }
}

void plot_data (udphist_t * ctx)
{
  unsigned nbin=256;
  float x[nbin];
  float re0[nbin];
  float im0[nbin];
  float re1[nbin];
  float im1[nbin];
  int ibin;
  int ichan, iant;

  float ymin = 0;
  float ymax_p0= 0;
  float ymax_p1 = 0;
  char all_zero = 1;

  cpgeras();

  for (ibin=0; ibin<256; ibin++)
  {
    x[ibin] = (float) ibin - 127;
    re0[ibin] = (float) ctx->histograms[0][ibin];
    im0[ibin] = (float) ctx->histograms[1][ibin];
    re1[ibin] = (float) ctx->histograms[2][ibin];
    im1[ibin] = (float) ctx->histograms[3][ibin];

    if (re0[ibin] > ymax_p0)
      ymax_p0 = re0[ibin];
    if (im0[ibin] > ymax_p0)
      ymax_p0 = im0[ibin];

    if (re1[ibin] > ymax_p1)
      ymax_p1 = re1[ibin];
    if (im1[ibin] > ymax_p1)
      ymax_p1 = im1[ibin];
  }
  cpgbbuf();
  cpgsci(1);

  ymax_p0 *= 1.1;
  ymax_p1 *= 1.1;

  // pol0 
  cpgswin(-130, 130, ymin, ymax_p0);
  cpgsvp(0.1, 0.9, 0.5, 0.9);
  cpgbox("BCST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("", "Count", "");

  cpgmtxt("T", -2.2, 0.05, 0.0, "Pol0");
  cpgsci(2);
  cpgslw(3);
  cpgbin (nbin, x, re0, 0);
  cpgsci(3);
  cpgbin (nbin, x, im0, 0);
  cpgslw(1);
  cpgsci(1);

  cpgsvp(0.1, 0.9, 0.1, 0.5);
  cpgbox("BCNST", 0.0, 0.0, "BCNST", 0.0, 0.0);
  cpglab("State", "Count", "");

  cpgswin(-130, 130, ymin, ymax_p1);

  // draw dotted line for the centre of the distribution
  cpgsls(2);
  cpgslw(2);
  cpgmove (0, 0);
  cpgdraw (0, ymax_p1);
  cpgsls(1);

  cpgmtxt("T", -2.2, 0.05, 0.0, "Pol1");
  cpgslw(3);
  cpgsci(2);
  cpgbin (nbin, x, re1, 0);
  cpgsci(3);
  cpgbin (nbin, x, im1, 0);
  cpgsci(1);
  cpgslw(1);

  cpgebuf();
}


