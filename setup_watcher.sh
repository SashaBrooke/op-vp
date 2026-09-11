# Setup the OPVP watcher service
sudo cp ./watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable watcher.service
sudo systemctl start watcher.service
