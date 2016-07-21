#!groovy

def katsdp = fileLoader.fromGit('jenkins/scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpinfrastructure', 'master', 'katpull', '')

katsdp.setDependencies(['ska-sa/katsdpsigproc/master', 'ska-sa/katsdpdockerbase/master'])

katsdp.commonBuild(maintainer: 'bmerry@ska.ac.za') {
    katsdp.stagePrepare()
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageMakeDocker()

    stage 'digitiser capture'
    node {
        deleteDir()
        unstash 'source'
        katsdp.makeDocker('katsdpingest_digitiser_capture', 'git/digitiser_capture')
    }
    // TODO: autotuning
}
