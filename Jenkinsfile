#!groovy

katsdp.setDependencies(['ska-sa/katsdpsigproc/test-jenkinsfile'])

def stageDigitiserCapture() {
    stage 'digitiser capture'
    node {
        deleteDir()
        unstash 'source'
        dir('git/digitiser_capture') {
            sh 'build-docker-image.sh katsdpingest_digitiser_capture'
        }
    }
}

katsdp.commonBuild(maintainer: 'bmerry@ska.ac.za') {
    katsdp.stagePrepare()
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageMakeDocker()
    stageDigitiserCapture()
    // TODO: autotuning
}
