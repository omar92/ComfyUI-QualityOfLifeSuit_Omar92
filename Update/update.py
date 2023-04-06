import importlib
from datetime import datetime

def pull(pygit2,repo, remote_name='origin', branch='master'):
    for remote in repo.remotes:
        #print("fetching latest changes: ",remote.name)
        if remote.name == remote_name:
            remote.fetch()
            remote_master_id = repo.lookup_reference('refs/remotes/origin/%s' % (branch)).target
            merge_result, _ = repo.merge_analysis(remote_master_id)
            # Up to date, do nothing
            if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
                return
            # We can just fastforward
            elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
                repo.checkout_tree(repo.get(remote_master_id))
                try:
                    master_ref = repo.lookup_reference('refs/heads/%s' % (branch))
                    master_ref.set_target(remote_master_id)
                except KeyError:
                    repo.create_branch(branch, repo.get(remote_master_id))
                repo.head.set_target(remote_master_id)
            elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
                repo.merge(remote_master_id)

                if repo.index.conflicts is not None:
                    for conflict in repo.index.conflicts:
                        print('Conflicts found in:', conflict[0].path)
                    raise AssertionError('Conflicts, ahhhhh!!')

                user = repo.default_signature
                tree = repo.index.write_tree()
                commit = repo.create_commit('HEAD',
                                            user,
                                            user,
                                            'Merge!',
                                            tree,
                                            [repo.head.target, remote_master_id])
                # We need to do this or git CLI will think we are still merging.
                repo.state_cleanup()
            else:
                raise AssertionError('Unknown merge analysis result')
def install_pygit2():
    # Helper function to install the pygit2 module if not already installed
    try:
        importlib.import_module('pygit2')
    except ImportError:
        import pip
        pip.main(['install', 'pygit2'])
def update(repoPath = "", branch_name="main" ):
    print(f"Updating: Quality of Life Suit...")

    install_pygit2()
    import pygit2

    repo = pygit2.Repository(repoPath)
    ident = pygit2.Signature('omar92', 'omar@92')
    try:
        #print("stashing current changes")
        repo.stash(ident)
    except KeyError:
        #print("nothing to stash")
        pass
    backup_branch_name = 'backup_branch_{}'.format(datetime.today().strftime('%Y-%m-%d_%H_%M_%S'))
    #print("creating backup branch: {}".format(backup_branch_name))
    repo.branches.local.create(backup_branch_name, repo.head.peel())

    #print(f"checking out {branch_name} branch")
    branch = repo.lookup_branch(str(branch_name))
    ref = repo.lookup_reference(branch.name)
    repo.checkout(ref)

    #print("pulling latest changes")
    pull(pygit2,repo, branch=branch_name)

    print(f"done: Quality of Life Suit, updated successfully...")

