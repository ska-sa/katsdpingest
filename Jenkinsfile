#! groovy

@Library('katsdpjenkins')

katsdp.setDependencies('ska-sa/katsdpdockerbase/master')
katsdp.standardBuild(docker_timeout:[time: 120, unit: "MINUTES']))
katsdp.mail('cschollar@ska.ac.za bmerry@ska.ac.za thomas@ska.ac.za')
