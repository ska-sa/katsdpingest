#!groovy

def katsdp = fileLoader.fromGit('scripts/katsdp.groovy', 'git@github.com:ska-sa/katsdpjenkins', 'master', 'katpull', '')

katsdp.setDependencies(['ska-sa/katsdpsigproc/master', 'ska-sa/katsdpdockerbase/master'])

katsdp.commonBuild(maintainer: 'bmerry@ska.ac.za') {
    katsdp.stagePrepare()
    katsdp.stageNosetestsGpu(cuda: true, opencl: true)
    katsdp.stageMakeDocker()

    if (currentBuild.result == null) {
        stage name: 'autotuning', concurrency: 1
        katsdp.simpleNode(label: 'cuda-GeForce_GTX_TITAN_X') {
            deleteDir()
            katsdp.unpackGit()
            katsdp.unpackVenv()
            katsdp.virtualenv('venv') {
                // TODO: update the script instead
                withEnv(["GIT_BRANCH=${env.BRANCH_NAME}"]) {
                    dir('git') {
                        sh './jenkins-autotune.sh titanx'
                    }
                }
            }
        }
    }

    stage 'digitiser capture'
    katsdp.simpleNode {
        deleteDir()
        katsdp.unpackGit()
        katsdp.makeDocker('katsdpingest_digitiser_capture', 'git/digitiser_capture')
    }
}
