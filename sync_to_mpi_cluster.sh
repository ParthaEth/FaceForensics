while inotifywait -r ../FaceForensics/*; do
  rsync -azhe "ssh -i ~/.ssh/id_rsa" ../FaceForensics/ pghosh@login.cluster.is.localnet:/home/pghosh/repos/never_eidt_from_cluster_remote_edit_loc/FaceForensics
done
