ps -eaf | grep spawn_main  | grep -v grep | awk '{print $2}' | xargs kill -s 9
ps -eaf | grep trainer  | grep -v grep | awk '{print $2}' | xargs kill -s 9
