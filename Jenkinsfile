#!groovy

katsdp.setDependencies(['ska-sa/katsdpsigproc/test-jenkinsfile'])

def stageDigitiserCapture() {
    stage 'digitiser capture'
    node {
        deleteDir()
        unstash 'source'
        katsdp.makeDocker('katsdpingest_digitiser_capture', 'git/digitiser_capture')
    }
}

katsdp.commonBuild(maintainer: 'bmerry@ska.ac.za') {
    katsdp.stagePrepare()
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageMakeDocker()
    stageDigitiserCapture()
    // TODO: autotuning
}
