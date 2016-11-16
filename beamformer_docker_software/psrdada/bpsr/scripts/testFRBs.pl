#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use Bpsr;           # Bpsr Module for configuration options
use strict;         # strict mode (like -Wall)
use Math::Trig;
use MIME::Lite;

use constant DL           => 1;
our %cfg = Bpsr::getConfig();
our %roach = Bpsr::getROACHConfig();

my ($cmd, $result, $response, $obs, $overview);
my @list = ();

our $results_dir = "/projects/p002_swin/superb";

chdir $results_dir;

if ($#ARGV == 0)
{
  push @list, $ARGV[0];
}
else
{
  $cmd = "find ".$results_dir." -maxdepth 1 -type d -name '20??-??-??-??:??:??' -printf '\%f\n' | sort -n";
  Dada::logMsg(2, DL, "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, DL, "main: ".$result." ".$response);

  @list = split(/\n/, $response);
}

foreach $obs ( @list )
{
  chdir $obs;

  Dada::logMsg(2, DL, "main: testing ".$obs);

  $cmd = "find -maxdepth 1 -type f -name '2*.cands_1024x768.png' | sort -n | tail -n 1";
  ($result, $response) = Dada::mySystem($cmd);
  if (($result eq "ok") && (-f $response))
  {
    Dada::logMsg(1, DL, "main: detectFRBs(".$obs.", ".$response.")");
    detectFRBs($obs, $response, "");
  }

  chdir $results_dir;
}

#
# detect FRBs from the candidate file in the current dir
#
sub detectFRBs($$$)
{
  my ($obs, $overview, $last_frb) = @_;

  my ($cmd, $result, $response);
  my ($source, $pid, $gl, $gb, $galactic_dm);

  my $to_email = 'ajameson@swin.edu.au';
  my $cc_email = 'ekeane@swin.edu.au';

  # get the SOURCE and PID
  $cmd = "grep ^SOURCE obs.info | awk '{print \$2}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $source) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$source);

  $cmd = "grep ^PID obs.info | awk '{print \$2}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $pid) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$pid);

  # extract the GL, GB from TCS
  $cmd = "grep \"beam_info beam='01'\" beaminfo.xml";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);

  my @parts = split(/ /, $response);
  if ($#parts != 5)
  {
    Dada::logMsg(2, DL, "detectFRBs: couldn't extract Gl Gb from beaminfo");
    return ("ok", "");
  }

  $gl = substr($parts[4], 4, (length($parts[4]) - 5));
  $gb = substr($parts[5], 4, (length($parts[5]) - 6));
  Dada::logMsg(2, DL, "detectFRBs: gl=".$gl." gb=".$gb);

  # determine the DM for this GL and GB
  $cmd = "cd \$HOME/opt/NE2001/runtime; ./NE2001 ".$gl." ".$gb." 100 -1 | grep ModelDM | awk '{print \$1}'";
  Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
  ($result, $galactic_dm) = Dada::mySystem($cmd);
  Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$galactic_dm);

  # determine the FRB detection command to run
  $cmd = "frb_detector.py -snr_cut 9 ".$galactic_dm;

  # check if any FRB's in candidates file
  Dada::logMsg(1, DL, "detectFRBs: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(1, DL, "detectFRBs: ".$result." ".$response);

  if (($result eq "ok") && ($response ne ""))
  {
    my @frbs = split(/\n/, $response);

    Dada::logMsg(2, DL, "detectFRBs: FRB Detection for ".$obs." [".($#frbs+1)." events]");
    if ($last_frb ne $obs)
    {
      $last_frb = $obs;
      my $subject = "Possible FRB in ".$obs;
      my $frb_name = "FRB".substr($obs, 2,2).substr($obs,5,2).substr($obs,8,2);

      # dont cc for multiple FRBs in a single observation (RFI)
      if ($#frbs > 10)
      {
        $cc_email = '';
      }

      my $msg = MIME::Lite->new(
        From    => 'BPSR FRB Detector <jam192@hipsr-srv0.atnf.csiro.au>',
        To      => $to_email,
        Cc      => $cc_email,
        Subject => 'New Detection: '.$frb_name,
        Type    => 'multipart/mixed',
        Data    => "Here's the PNG file you wanted"
      );

      # generate HTML part of email
      my $html = "<body>";

      $html .= "<table cellpadding=2 cellspacing=2>\n";

      $html .= "<tr><th style='text-align: left;'>UTC START</th><td>".$obs."</td></tr>\n";
      $html .= "<tr><th style='text-align: left;'>Source</th><td>".$source."</td></tr>\n";
      $html .= "<tr><th style='text-align: left; padding-right:5px;'>PID</th><td>".$pid."</td></tr>\n";
      $html .= "<tr><th style='text-align: left;'>NE2001 DM</th><td>".$galactic_dm."</td></tr>\n";

      $html .= "</table>\n";

      # contact all the dbevent processes and dump the raw data
      my @dbevent_socks = ();
      my ($host, $port, $sock, $frb, $beam);
      my ($time_secs, $time_subsecs, $filter_time, $smearing_time, $total_time, $start_time, $end_time);
      my ($start_utc, $end_utc, $frb_event_string);
      my $event_sock = 0;

      # TODO remove this manual disable
      if ($event_sock)
      {
        my $i = 0;
        for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
        {
          $host = $cfg{"PWC_".$i};
          $port = int($cfg{"CLIENT_EVENT_BASEPORT"}) + int($i);
          Dada::logMsg(2, DL, "detectFRBs: opening connection for FRB dump to ".$host.":".$port);
          $sock = Dada::connectToMachine($host, $port);
          if ($sock)
          {
            Dada::logMsg(3, DL, "detectFRBs: connection to ".$host.":".$port." established");
            push @dbevent_socks, $sock;
          }
          else
          {
            Dada::logMsg(0, DL, "detectFRBs: connection to ".$host.":".$port." failed");
          }
        }
      }

      my $src_html = "<h3>Beams Positions &amp; Known Sources</h3>";

      # setup a hash of psrs in beams
      my %psrs_in_beams = ();

      # firstly check if there is a known PSR in this position source in the 
      if (-f "beaminfo.xml")
      {
        $src_html .= "<table width='100%' border=0 cellpadding=2px cellspacing=2px>\n";
        $src_html .= "<tr>";
        $src_html .= "<th style='text-align: left;'>Beam</th>";
        $src_html .= "<th style='text-align: left;'>RA</th>";
        $src_html .= "<th style='text-align: left;'>DEC</th>";
        $src_html .= "<th style='text-align: left;'>Gl</th>";
        $src_html .= "<th style='text-align: left;'>Gb</th>";
        $src_html .= "<th style='text-align: left;'>PSR</th>";
        $src_html .= "</tr>\n";

        open FH, "<beaminfo.xml";
        my @lines = <FH>;
        close FH;

        my ($line, $ra, $dec, $gl, $gb, $psrs, $name, $dm);
        my $beam = "00";
        foreach $line (@lines)
        {
          chomp $line;
          if ($line =~ m/beam_info beam/)
          {
            $gl = "--";
            $gb = "--";
            $psrs = "";
            my @bits = split(/ /,$line);
            $beam = substr($bits[1], 6, 2);
            $ra = substr($bits[2],5,length($bits[2])-6);
            $dec = substr($bits[3],6,length($bits[3])-7);
            if ($#bits > 3)
            {
              $gl = substr($bits[4],4,length($bits[4])-6);
              $gb = substr($bits[5],4,length($bits[5])-6);
            }
            $psrs_in_beams{$beam} = ();
          } 
          elsif ($line =~ m/beam_info/)  
          {
            $src_html .= "<tr>";
            $src_html .=   "<td>".$beam."</td>";
            $src_html .=   "<td>".$ra."</td>";
            $src_html .=   "<td>".$dec."</td>";
            $src_html .=   "<td>".$gl."</td>";
            $src_html .=   "<td>".$gb."</td>";
            $src_html .=   "<td>".$psrs."</td>";
            $src_html .= "</tr>";
          }
          elsif ($line =~ m/psr name/)
          {
            my @bits = split(/ /,$line);
            $name = substr($bits[1], 6, length($bits[1])-7);
            $dm = substr($bits[2], 4, length($bits[2])-5);
            if ($psrs ne "")
            {
              $psrs .= ", ";
            }
            $psrs .= $name." [DM=".$dm."]\n";
            $psrs_in_beams{$beam}{$dm} = $name;
          }
          else
          {
            Dada::logMsg(3, DL, "detectFRBs: ignoring");
          }
        }
        $src_html .= "</table>";
      }
      else
      {
        $src_html .= "<p>No beaminfo.xml existed for this observation</p>\n";
      }

      $html .= "<hr/>\n";

      #
      # FRB Table
      #
      $html .= "<h3>FRB Detections</h3>";
      $html .= "<table width='100%' border=0 cellpadding=2px cellspacing=2px>\n";
      $html .= "<tr>";
      $html .= "<th style='text-align: left;'>SNR</th>";
      $html .= "<th style='text-align: left;'>Time</th>";
      #$html .= "<th style='text-align: left;'>Sample</th>";
      $html .= "<th style='text-align: left;'>DM</th>";
      $html .= "<th style='text-align: left;'>Length</th>";
      $html .= "<th style='text-align: left;'>Beam</th>";
      $html .= "<th style='text-align: left;'>Known Source(s)</th>";
      $html .= "</tr>\n";

      my $num_frbs_legit = 0;

      # NB this requires a prefix, see below
      $frb_event_string .= $obs."\n";
      $frb_event_string .= "# START_UTC   STOP_UTC   DM   SNR\n";

      # a record of good vs bad
      my %frbs_legit = ();

      foreach $frb (@frbs)
      {
        Dada::logMsg(2, DL, "detectFRBs: frb=".$frb);
        my ($snr, $time, $sample, $dm, $filter, $prim_beam) = split(/\t/, $frb);
        my ($delta_dm, $psr_dm, $related_psrs, $padded_prim_beam, $legit);

        $related_psrs = "";
        $delta_dm = $dm * 0.05;
        $padded_prim_beam = sprintf("%02d", $prim_beam);
        $legit = 1;

        # check if there is a known source nearby this DM for this beam
        foreach $psr_dm ( keys %{ $psrs_in_beams{$padded_prim_beam} } )
        {
          # always include any pulsars in the beam
          $related_psrs .= $psrs_in_beams{$padded_prim_beam}{$psr_dm}." [DM=".$psr_dm."]<br/>";

          Dada::logMsg(2, DL, "detectFRBs: FRB dm=".$dm." testing psr_dm=".$psr_dm);
          # the window of DM that we consider the source is a match for the FRB dm
          if ($dm < ($psr_dm + $delta_dm))
          {
            Dada::logMsg(1, DL, "detectFRBs: ignoring event since dm=".$dm." too close to ".$psrs_in_beams{$padded_prim_beam}{$psr_dm}." [DM=".$psr_dm."]");
            $legit = 0;
          }
        }

        $frbs_legit{$frb} = $legit;

        if ($legit)
        {
          # we have a least 1 FRB in this list
          $num_frbs_legit++;

          $filter_time   = (2 ** $filter) * 0.000064;

          $html .= "<tr>";
          $html .= "<td>".$snr."</td>";
          $html .= "<td>".sprintf("%5.2f",$time)."</td>";
          #$html .= "<td>".$sample."</td>";
          $html .= "<td>".$dm."</td>";
          $html .= "<td>".($filter_time * 1000)."</td>";
          $html .= "<td>".$padded_prim_beam."</td>";
          $html .= "<td>".$related_psrs."</td>";
          $html .= "</tr>\n";

          $cmd = "dmsmear -f 1382 -b 400 -n 1024 -d ".$dm." -q";
          Dada::logMsg(2, DL, "detectFRBs: cmd=".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
          $smearing_time = $response;

          $total_time = $filter_time + $smearing_time;

          # be generous and allow 2 times the smearing time before and after the event
          $start_time = $time - (2 * $total_time);
          $end_time   = $time + (3 * $total_time);

          Dada::logMsg(2, DL, "detectFRBs: FRB filter_time=".$filter_time." smearing_time=".$smearing_time." total_time=".$total_time);
          Dada::logMsg(2, DL, "detectFRBs: FRB start_time=".$start_time." end_time=".$end_time);

          # determine abosulte start time in UTC
          $start_utc = Dada::addToTimeFractional($obs, $start_time);

          # determine abosulte end time in UTC
          $end_utc = Dada::addToTimeFractional($obs, $end_time);

          $frb_event_string .= $start_utc." ".$end_utc." ".$dm." ".$snr."\n";

          Dada::logMsg(1, DL, "detectFRBs: FRB frb_string=".$start_utc." ".$end_utc." ".$dm." ".$snr);
        }
      }
      $html .= "  </table>";

      $html .= "<hr/>";

      $html .= $src_html;

      # now only do something if we have at least 1 legitimate FRB
      if ($num_frbs_legit > 0)
      {
        Dada::logMsg(1, DL, "detectFRBs: EMAIL FRB Detection for ".$obs);

        # first send the number of events to each socket
        $frb_event_string = "N_EVENTS ".($#frbs + 1)."\n".$frb_event_string;

        Dada::logMsg(1, DL, "detectFRBs: sending FRB Event dump to ".($#dbevent_socks+1)." sockets");
        print $frb_event_string;

        # TODO remove this manual disable
        if ($event_sock)
        {
          my $i = 0;
          for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
          {
            $host = $cfg{"PWC_".$i};
            $port = int($cfg{"CLIENT_EVENT_BASEPORT"}) + int($i);
            Dada::logMsg(2, DL, "detectFRBs: opening connection for FRB dump to ".$host.":".$port);
            $sock = Dada::connectToMachine($host, $port);
            if ($sock)
            {
              Dada::logMsg(3, DL, "detectFRBs: connection to ".$host.":".$port." established");
              push @dbevent_socks, $sock;
            }
            else
            {
              Dada::logMsg(0, DL, "detectFRBs: connection to ".$host.":".$port." failed");
            }
          }
        }

        # tell the listening dbevent process to dump the message
        foreach $sock (@dbevent_socks)
        {
          print $sock $frb_event_string;
          close ($sock);
          $sock = 0;
        }

        if ($num_frbs_legit > 5)
        {
          $html .= "<p>NOTE: ".($num_frbs_legit)." FRBs were detected. This execeeded the ".
                   " limit of 5, so none of them were plotted</p>";
        }

        $html .= "<hr/>";
        $html .= "<h3>Plots</h3>";
        $html .= "</body>";

        ### Add the html message part:
        $msg->attach(
          Type     => 'text/html',
          Data     => $html,
        );

        $cmd = "find -maxdepth 1 -type f -name '2*.cands_1024x768.png' | sort -n | tail -n 1";
        Dada::logMsg(2, DL, "detectFRBs: cmd=".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
        if ($result eq "ok")
        {
          ### Attach a part... the make the message a multipart automatically:
          $msg->attach(
            Type        => 'image/png',
            Id          => 'heimdall_overview',
            Path        => $results_dir.'/'.$obs.'/'.$response,
            Filename    => $frb_name.'.png',
            Disposition => 'attachment'
          );
        }

        # if we have more than 5 FRB events, dont plot them all
        #if ($num_frbs_legit <= 5)
        if (0)
        {
          foreach $frb (@frbs)
          {
            if ($frbs_legit{$frb})
            {
              Dada::logMsg(2, DL, "detectFRBs: FRB=".$frb);
              my ($snr, $time, $sample, $dm, $filter, $prim_beam) = split(/\t/, $frb);

              # determine host for the beam
              my $i=0;
              $host = "hipsr7";
              my $beam = sprintf("%02d", $prim_beam);

              my $fil_file = "/nfs/raid0/bpsr/perm/".$pid."/".$obs."/".$beam."/".$obs.".fil";
              my $plot_cmd = "trans_freqplus_plot.py ".$fil_file." ".$sample." ".$dm." ".$filter." ".$snr;
              my $local_img = $obs."_".$beam."_".$sample.".png";

              # create a freq_plus file
              $cmd = "ssh bpsr@".$host." '".$plot_cmd."' > /tmp/".$local_img;
              Dada::logMsg(1, DL, "detectFRBs: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              Dada::logMsg(2, DL, "detectFRBs: ".$result." ".$response);

              # get the first 5 chars / bytes of the file 
              $cmd = "head -c 5 /tmp/".$local_img;
              Dada::logMsg(1, DL, "detectFRBs: ".$cmd);
              ($result, $response) = Dada::mySystem($cmd);
              Dada::logMsg(2, DL, "detectFRBs: ".$result." ".$response);

              # if we did manage to find the filterbank file
              if (($result eq "ok") && ($response ne "ERROR"))
              {
                $msg->attach(
                  Type        => 'image/png',
                  Path        => '/tmp/'.$local_img,
                  Filename    => $local_img,
                  Disposition => 'attachment'
                );
              }
            }
          }
        }

        $msg->send;

        $cmd = "rm -f /tmp/*_??_*.png";
        Dada::logMsg(2, DL, "detectFRBs: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(3, DL, "detectFRBs: ".$result." ".$response);
      }
    }
    else
    {
      Dada::logMsg(2, DL, "detectFRBs: last_frb=".$last_frb." obs=".$obs);
    }
  }
  else
  {
    Dada::logMsg(2, DL, "detectFRBs: NO FRB Detection for ".$obs);
    $last_frb = "";
  }

  return ("ok", $last_frb);

}

