# Connecting OneDrive to Google Colab on Windows

## 1. Download the rclone application from https://rclone.org/downloads/ and select your version and CPU Architecture (Intel/AMD - 64 Bit works for me)

## 2. Install the application - pretty straight forward

## 3. Open the Terminal, navigate to the directory where rclone.exe is located and run `.\rclone authorize "onedrive"`. NOTE: If you are not used to Terminal like me, after opening the Terminal, type `cd`, navigate to the directory on the Windows GUI and drag the folder into the Terminal, then write `cd` at the beginning. A beginners guide on Terminal can be found here: https://missing.csail.mit.edu/2020/

## 4. The result of the authorization code will prompt you to login to your OneDrive account

## 5. On successful login, navigate back to the Terminal and you will see the access code. Copy the access code and paste in the Google Colab cell (you will use it later so copying it there will become handy).

## 6. Run



# Download rclone
!wget https://downloads.rclone.org/v1.70.3/rclone-v1.70.3-linux-amd64.deb
!apt install ./rclone-v1.70.3-linux-amd64.deb

# Run the configuration
!rclone config

!sudo mkdir /content/onedrive

!sudo apt update
!sudo apt install fuse3

!nohup rclone mount onedrive: /content/onedrive --vfs-cache-mode writes &