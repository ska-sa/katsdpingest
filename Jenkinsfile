#!groovy

@Library('katsdpjenkins') _

katsdp.setDependencies( 'ska-sa/katsdpdockerbase/master',
                        'ska-sa/katsdpsigproc/master',
                        'ska-sa/katsdptelstate/maseter')
katsdp.standardBuild(docker_timeout:[time: 120, unit: "MINUTES']))
katsdp.mail('cschollar@ska.ac.za bmerry@ska.ac.za')
