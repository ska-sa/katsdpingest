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
#include <fftw3.h>

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
  unsigned int nfft;

  // number of output channels
  unsigned int nchan_out;

  // number of dimensions [should always be 2]
  unsigned int ndim;

  // which polarisations to plot 
  int polarisation;

  // size of the UDP packet
  unsigned int resolution;

  unsigned int verbose;

  float * x_points;

  float ** y_points;

  float * x_ts;

  float ** y_ts;

  float *** fft_in;     // input timeseries

  float ** fft_out;     // output spectra (oversampled)

  float ** fft_in2;     // input spectra (critically sampled)

  float ** fft_out2;    // output timeseries (critically sampled)

  float ** fft_in3;     // input timeseries (critically sampled)

  float ** fft_out3;    // output spectra (critically sampled)

  unsigned int fft_count;

  uint64_t num_integrated;

  uint64_t to_integrate;

  unsigned int plot_log;

  float ymin;

  float ymax;

  float base_freq;

  float bw;

  int zap_dc;

  float xmin;

  float xmax;

  fftwf_plan plan;

  unsigned dsb;

} udpplot_t;

int udpplot_init (udpplot_t * ctx);
int udpplot_prepare (udpplot_t * ctx);
int udpplot_destroy (udpplot_t * ctx);

void append_samples (udpplot_t * ctx, void * buffer, uint64_t isamp,  unsigned npt, uint64_t nbytes);
void detect_data (udpplot_t * ctx);
void fft_data (udpplot_t * ctx);
void plot_data (udpplot_t * ctx);
void plot_ts (udpplot_t * ctx, unsigned channel);

void usage ();


void usage()
{
  fprintf (stdout,
     "dadafftplot [options] dadafile\n"
     " -F min,max  set the min,max x-value (e.g. frequency zoom)\n" 
     " -l          plot logarithmically\n"
     " -n npt      number of points in each coarse channel fft [default 1024]\n"
     " -D device   pgplot device name\n"
     " -t num      number of FFTs to avaerage into each plot\n"
     " -v          be verbose\n");
}

int main (int argc, char **argv)
{
  // flag set in verbose mode
  unsigned int verbose = 0;

  // PGPLOT device name
  char * device = "/xs";

  int arg = 0;

  unsigned int nfft = 1024;

  unsigned int plot_log = 0;

  float xmin = 0;
  float xmax = 0;
  float ymin = FLT_MAX;
  float ymax = -FLT_MAX;

  float base_freq;
  unsigned int to_integrate = 8;
  unsigned zap_dc = 0;

  while ((arg=getopt(argc,argv,"D:F:ln:t:vz")) != -1)
  {
    switch (arg)
    {
      case 'D':
        device = strdup(optarg);
        break;

      case 'F':
      {
        if (sscanf (optarg, "%f,%f", &xmin, &xmax) != 2)
        {
          fprintf (stderr, "could not parse xrange from %s\n", optarg);
          return (EXIT_FAILURE);
        }
        break;
      }

      case 'l':
        plot_log = 1;
        break; 

      case 'n':
        nfft = atoi(optarg);
        break;

      case 't':
        to_integrate = atoi (optarg);
        break;

      case 'v':
        verbose++;
        break;

      case 'z':
        zap_dc  = 1;
        break;

      default:
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

  //size_t pkt_size = 2560;
  //void * pkt = (void *) malloc (pkt_size);

  void * raw = malloc (data_size);
  bytes_read = read (fd, raw, data_size);
  if (verbose)
    fprintf (stderr, "read %lu bytes\n", bytes_read);

  udpplot_t udpplot;
  udpplot.verbose = verbose;
  udpplot.ndim = 2;
  udpplot.npol = 2;


  if (ascii_header_get(header, "NCHAN", "%d", &(udpplot.nchan_in)) != 1)
  {
    fprintf (stderr, "could not extract NCHAN from header\n");
    return EXIT_FAILURE;
  }
  udpplot.nfft = nfft;
  udpplot.nchan_out = udpplot.nchan_in * nfft;

  float cfreq;
  if (ascii_header_get(header, "FREQ", "%f", &cfreq) != 1)
  {
    fprintf (stderr, "could not extract FREQ from header\n");
    return EXIT_FAILURE;
  }

  if (ascii_header_get(header, "BW", "%f", &(udpplot.bw)) != 1)
  {     
    fprintf (stderr, "could not extract BW from header\n");
    return EXIT_FAILURE; 
  }

  if (ascii_header_get(header, "DSB", "%u", &(udpplot.dsb)) != 1)
  {
    fprintf (stderr, "could not extract DSB from header\n");
    return EXIT_FAILURE;
  }

  if (verbose)
    fprintf (stderr, "main: DSB=%u\n", udpplot.dsb);

  if (header)
    free(header);
  header = 0;

#if 0
  if (verbose > 1)
  {
    fprintf (stderr, "[T][F][S] (re, im)\n");
    int8_t * ptr = (int8_t *) raw;
    unsigned ipol, isamp, ichan;
    for (isamp=0; isamp<4; isamp++)
    {
      for (ichan=0; ichan<udpplot.nchan_in; ichan++)
      {
        for (ipol=0; ipol<udpplot.nant; ipol++)
        {
          fprintf (stderr, "[%d][%d][%d] (%d, %d)\n", isamp, ichan, ipol, ptr[0], ptr[1]);
          ptr += 2;     
        }
      }
    }
  }
#endif

  udpplot.base_freq = cfreq - (udpplot.bw / 2);
  if ((xmin == 0) && (xmax == 0))
  {
    xmin = udpplot.base_freq;
    xmax = udpplot.base_freq + udpplot.bw;
  }

  udpplot.polarisation = -1;

  udpplot.zap_dc = zap_dc;
  udpplot.num_integrated = 0;
  udpplot.fft_count = 0;
  udpplot.to_integrate = to_integrate;

  udpplot.plot_log = plot_log;
  udpplot.xmin = xmin;
  udpplot.xmax = xmax;
  udpplot.ymin = ymin;
  udpplot.ymax = ymax;

  if (verbose)
    fprintf (stderr, "Freq range: %f - %f MHz\n", udpplot.xmin, udpplot.xmax);

  if (verbose)
    fprintf(stderr, "mopsr_dadafftplot: using device %s\n", device);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "mopsr_dadafftplot: error opening plot device\n");
    exit(1);
  }
  cpgask(1);

  udpplot_t * ctx = &udpplot;

  // allocate require resources
  if (udpplot_init (ctx) < 0)
  {
    fprintf (stderr, "ERROR: Could not alloc memory\n");
    exit(1);
  }

  // cloear packets ready for capture
  udpplot_reset (ctx);
  const unsigned ndim = 2;
  const unsigned npol = 2;

  uint64_t isample = 0;
  uint64_t nsamples = data_size / (ndim * npol * ctx->nchan_in);
  uint64_t nsamples_per_append = ctx->nfft;

  while (isample < nsamples)
  {
    fprintf (stderr, "appending isamp=%lu\n", isample);
    append_samples (ctx, raw, isample, ctx->nfft, data_size);
    isample += ctx->nfft;

    fft_data (ctx);
    detect_data (ctx);
    ctx->num_integrated ++;

    if (ctx->num_integrated >= ctx->to_integrate)
    {
      if (verbose)
        fprintf(stderr, "plotting %d spectra (%d pts) in %d channels\n", 
                ctx->num_integrated, ctx->nfft * ctx->num_integrated, 
                ctx->nchan_out);
      //plot_ts(ctx, 263);
      plot_data (ctx);
      udpplot_reset (ctx);

      sleep (1);
    }
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
  unsigned ichan;
  float mhz_per_out_chan = ctx->bw / (float) ctx->nchan_out;
  for (ichan=0; ichan < ctx->nchan_out; ichan++)
  {
    ctx->x_points[ichan] = ctx->base_freq + (((float) ichan) * mhz_per_out_chan);
  }

  unsigned ipol;
  unsigned ifft;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan_out; ichan++)
    {
      ctx->y_points[ipol][ichan] = 0;
      ctx->fft_out[ipol][2*ichan+0] = 0;
      ctx->fft_out[ipol][2*ichan+1] = 0;
    }
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      for (ifft=0; ifft < ctx->nfft; ifft++)
      {
        ctx->fft_in[ipol][ichan][2*ifft+0] = 0;
        ctx->fft_in[ipol][ichan][2*ifft+1] = 0;
      }
    }
  }
  ctx->num_integrated = 0;
  ctx->fft_count = 0;
}

int udpplot_destroy (udpplot_t * ctx)
{

  fftwf_destroy_plan (ctx->plan);
  unsigned int ipol;
  unsigned int ichan;
  for (ipol=0; ipol<ctx->npol; ipol++)
  {
    if (ctx->y_points[ipol])
      free(ctx->y_points[ipol]);
    ctx->y_points[ipol] = 0;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      if (ctx->fft_in[ipol][ichan])
        free (ctx->fft_in[ipol][ichan]);
      ctx->fft_in[ipol][ichan] = 0;
    }
    if (ctx->fft_in[ipol])
      free (ctx->fft_in[ipol]);
    ctx->fft_in[ipol] = 0;
    if (ctx->fft_out[ipol])
      free (ctx->fft_out[ipol]);
    ctx->fft_out[ipol] = 0;
  }

  if (ctx->fft_in)
    free (ctx->fft_in);
  ctx->fft_in = 0;

  if (ctx->fft_out)
    free (ctx->fft_out);
  ctx->fft_out = 0;

  if (ctx->y_points)
    free(ctx->y_points);
  ctx->y_points = 0;

  if (ctx->x_points)
    free(ctx->x_points);
  ctx->x_points = 0;

}

int udpplot_init (udpplot_t * ctx)
{
  if (ctx->verbose > 1)
    fprintf(stderr, "mopsr_udpdb_init_receiver()\n");

  ctx->x_points = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_points = (float **) malloc(sizeof(float *) * ctx->npol);
  ctx->x_ts = (float *) malloc (sizeof(float) * ctx->nchan_out);
  ctx->y_ts = (float **) malloc(sizeof(float *) * ctx->ndim);
  ctx->fft_in = (float ***) malloc(sizeof(float **) * ctx->npol);
  ctx->fft_out = (float **) malloc(sizeof(float *) * ctx->npol);

  ctx->y_ts[0] = (float *) malloc(sizeof(float) * ctx->nfft);
  ctx->y_ts[1] = (float *) malloc(sizeof(float) * ctx->nfft);
  unsigned ipt;
  for (ipt=0; ipt< ctx->nfft; ipt++)
  {
    ctx->x_ts[ipt] = ipt;
  }

  unsigned int ipol;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    ctx->y_points[ipol] = (float *) malloc (sizeof(float) * ctx->nchan_out);
    ctx->fft_in[ipol] = (float **) malloc (sizeof(float *) * ctx->nchan_in);
    ctx->fft_out[ipol] = (float *) malloc (sizeof(float) * ctx->nchan_out * 2);

    unsigned int ichan;
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      ctx->fft_in[ipol][ichan] = (float *) malloc (sizeof(float) * ctx->nfft * 2);
    }
  }

  float * input  = (float *) ctx->fft_in[0][0];
  fftwf_complex * output = (fftwf_complex *) ctx->fft_out[0];

  CHECK_ALIGN(input);
  CHECK_ALIGN(output);

  int direction_flags = FFTW_FORWARD;
  int flags = 0;

  ctx->plan = fftwf_plan_dft_r2c_1d (ctx->nfft, input, output, flags);

  return 0;
}

// copy data from packet in the fft input buffer
void append_samples (udpplot_t * ctx, void * buffer, uint64_t isamp,  unsigned npt, uint64_t nbytes)
{
  unsigned ichan, ipol, ipt, iheap, idat;

  //unsigned nheap = nbytes / (ctx->nchan_in * ctx->npol * ctx->ndim * 256);
  unsigned nheap = npt / 256;

  size_t offset = isamp * ctx->nchan_in * ctx->npol * ctx->ndim;
  int8_t * in = ((int8_t *) buffer) + offset;

  for (iheap=0; iheap<nheap; iheap++)
  {
    for (ipol=0; ipol < ctx->npol; ipol++)
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        for (idat=0; idat<256; idat++)
        {
          if (ichan == 263)
          {
            ctx->fft_in[ipol][ichan][(2*(ipt+idat)) + 0] = (float) in[0];
            ctx->fft_in[ipol][ichan][(2*(ipt+idat)) + 1] = (float) in[1];
          }
          else
          {
            ctx->fft_in[ipol][ichan][(2*(ipt+idat)) + 0] = (float) 0;
            ctx->fft_in[ipol][ichan][(2*(ipt+idat)) + 1] = (float) 0;
          }

          if (ichan == 263)
          {
            ctx->y_ts[0][ipt+idat] = (float) in[0];
            ctx->y_ts[1][ipt+idat] = (float) in[1];
          }

          in += 2;
        }
      }
    }
    ipt += 256;
  }
}

void fft_data (udpplot_t * ctx)
{
  unsigned int ipol, ichan, ipt;
  float * src;
  float * dest;

  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    for (ichan=0; ichan < ctx->nchan_in; ichan++)
    {
      src = ctx->fft_in[ipol][ichan];
      dest = ctx->fft_out[ipol] + (ichan * ctx->nfft * 2);

      fftwf_execute_dft_r2c (ctx->plan, (float *) src, (fftwf_complex*) dest);
    }
  }
}

void detect_data (udpplot_t * ctx)
{
  unsigned ipol = 0;
  unsigned ichan = 0;
  unsigned ibit = 0;
  unsigned halfbit = ctx->nfft / 2;
  unsigned offset = 0;
  unsigned basechan = 0;
  unsigned newchan;
  unsigned shift;
  float a, b;

  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    if ((ctx->polarisation < 0) || (ctx->polarisation == ipol))
    {
      for (ichan=0; ichan < ctx->nchan_in; ichan++)
      {
        offset = (ichan * ctx->nfft * 2);
        basechan = ichan * ctx->nfft;

        if (ctx->dsb == 1)
        {
          // first half [flipped - A]
          for (ibit=halfbit; ibit<ctx->nfft; ibit++)
          {
            a = ctx->fft_out[ipol][offset + (ibit*2) + 0];
            b = ctx->fft_out[ipol][offset + (ibit*2) + 1];
            newchan = (ibit-halfbit);
            ctx->y_points[ipol][basechan + newchan] += ((a*a) + (b*b));
          }

          // second half [B]
          for (ibit=0; ibit<halfbit; ibit++)
          {
            a = ctx->fft_out[ipol][offset + (ibit*2) + 0];
            b = ctx->fft_out[ipol][offset + (ibit*2) + 1];
            newchan = (ibit+halfbit);
            ctx->y_points[ipol][basechan + newchan] += ((a*a) + (b*b));
          }
        } else {
          for (ibit=0; ibit<ctx->nfft; ibit++)
          {
            a = ctx->fft_out[ipol][offset + (ibit*2) + 0];
            b = ctx->fft_out[ipol][offset + (ibit*2) + 1];
            ctx->y_points[ipol][basechan + ibit] += ((a*a) + (b*b));
          }
        }
        if (ctx->zap_dc && ichan == 0)
          ctx->y_points[ipol][ichan] = 0;
      }
    }
  }
}


void plot_data (udpplot_t * ctx)
{
  if (ctx->verbose)
    fprintf(stderr, "plot_packet()\n");

  int ichan = 0;
  unsigned ipol = 0;
  unsigned iframe = 0;
  float ymin = ctx->ymin;
  float ymax = ctx->ymax;

  int xchan_min = -1;
  int xchan_max = -1;

  // determined channel ranges for the x limits
  for (ichan=0; ichan < ctx->nchan_out; ichan++)
  {
    if ((xchan_min == -1) && (ctx->x_points[ichan] >= ctx->xmin))
      xchan_min = ichan;
  }
  for (ichan=(ctx->nchan_out-1); ichan > 0; ichan--)
  {
    if ((xchan_max == -1) && (ctx->x_points[ichan] <= ctx->xmax))
      xchan_max = ichan;
  }

  // calculate limits
  if ((ctx->ymin == FLT_MAX) && (ctx->ymax == -FLT_MAX))
  {
    for (ipol=0; ipol < ctx->npol; ipol++)
    {
      if ((ctx->polarisation < 0) || (ctx->polarisation == ipol))
      {
        for (ichan=0; ichan < ctx->nchan_out; ichan++)
        {
          if (ctx->plot_log)
            ctx->y_points[ipol][ichan] = (ctx->y_points[ipol][ichan] > 0) ? log10(ctx->y_points[ipol][ichan]) : 0;
          if ((ichan > xchan_min) && (ichan < xchan_max))
          {
            if (ctx->y_points[ipol][ichan] > ymax) ymax = ctx->y_points[ipol][ichan];
            if (ctx->y_points[ipol][ichan] < ymin) ymin = ctx->y_points[ipol][ichan];
          }
        }
      }
    }
  }
  if (ctx->verbose)
  {
    fprintf(stderr, "plot_packet: ctx->xmin=%f, ctx->xmax=%f\n", ctx->xmin, ctx->xmax);
    fprintf(stderr, "plot_packet: ymin=%f, ymax=%f\n", ymin, ymax);
  }

  cpgbbuf();
  cpgsci(1);
  if (ctx->plot_log)
  {
    cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 20);
    cpglab("Channel", "log\\d10\\u(Power)", "Bandpass");
  }
  else
  {
    cpgenv(ctx->xmin, ctx->xmax, ymin, ymax, 0, 0);
    cpglab("Channel", "Power", "Bandpass");
  }

  float line_x[2];
  float line_y[2];
  float percent_chan;
  float ifreq;

  float oversampling_difference = ((5.0 / 32.0) * (ctx->bw / ctx->nchan_in)) / 2.0;
  cpgsls(2);
  for (ichan=0; ichan < ctx->nchan_in; ichan++)
  {
    line_y[0] = ymin;
    line_y[1] = ymin + (ymax - ymin);

    percent_chan = (float) ichan / (float) ctx->nchan_in;
    percent_chan *= ctx->bw;
    
    ifreq = ctx->base_freq + percent_chan;
    line_x[0] = line_x[1] = ifreq;
    cpgline(2, line_x, line_y);

    line_y[0] = ymin;
    line_y[1] = ymin + (ymax - ymin) / 4;

    line_x[0] = line_x[1] = ifreq - oversampling_difference;
    cpgline(2, line_x, line_y);

    line_x[0] = line_x[1] = ifreq + oversampling_difference;
    cpgline(2, line_x, line_y);
  }
  cpgsls(1);

  char ant_label[10];
  int ant_id = 0;
  for (ipol=0; ipol < ctx->npol; ipol++)
  {
    if ((ctx->polarisation < 0) || (ctx->polarisation == ipol))
    {
      //sprintf(ant_label, "Ant %d", ipol);
      cpgsci(ipol + 2);
      //cpgmtxt("T", 0.5 + (0.9 * ipol), 0.0, 0.0, ant_label);
      cpgline(ctx->nchan_out, ctx->x_points, ctx->y_points[ipol]);
    }
  }
  cpgebuf();
}

void plot_ts (udpplot_t * ctx, unsigned channel)
{
  if (ctx->verbose)
    fprintf(stderr, "plot_ts()\n");

  float xmin = 0;
  float xmax = (float) ctx->nfft;
  float ymin = -140;
  float ymax = 140;

  cpgbbuf();
  cpgsci(1);

  cpgenv(xmin, xmax, ymin, ymax, 0, 0);
  cpglab("Sample", "Voltage", "Time Series");

  cpgsci(2);
  //cpgline(ctx->nfft, ctx->x_ts, ctx->y_ts[0]);
  cpgslw(3);
  cpgpt(ctx->nfft, ctx->x_ts, ctx->y_ts[0], 1);
  cpgsci(3);
  cpgpt(ctx->nfft, ctx->x_ts, ctx->y_ts[1], 1);
  //cpgline(ctx->nfft, ctx->x_ts, ctx->y_ts[1]);
  cpgslw(1);
  cpgebuf();
}

