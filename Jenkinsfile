#!groovy

@Library('katsdpjenkins') _

katsdp.setDependencies([
    'ska-sa/katsdpsigproc/master',
    'ska-sa/katsdpdata/master',
    'ska-sa/katsdpdockerbase/master',
    'ska-sa/katsdpservices/master',
    'ska-sa/katsdptelstate/master'])

katsdp.standardBuild(subdir: 'katsdpcam2telstate')

catchError {
    katsdp.stagePrepare(timeout: [time: 60, unit: 'MINUTES'])
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageMakeDocker(venv: true)

    stage('autotuning') {
        if (currentBuild.result == null) {
            katsdp.simpleNode(label: 'cuda-GeForce_GTX_TITAN_X') {
                deleteDir()
                katsdp.unpackGit()
                katsdp.unpackVenv()
                katsdp.unpackKatsdpdockerbase()
                katsdp.virtualenv('venv') {
                    dir('git') {
                        lock("katsdpingest-autotune-${env.BRANCH_NAME}") {
                            sh './jenkins-autotune.sh titanx'
                        }
                    }
                }
            }
        }
    }

    stage('digitiser capture') {
        katsdp.simpleNode {
            deleteDir()
            katsdp.unpackGit()
            katsdp.unpackKatsdpdockerbase()
            katsdp.makeDocker('katsdpingest_digitiser_capture', 'git/digitiser_capture')
        }
    }
}
katsdp.mail('bmerry@ska.ac.za')
