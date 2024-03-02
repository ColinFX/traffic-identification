# srsLTE Installation Guide

## References

* [srsRAN Official Documentation](https://docs.srsran.com/projects/project/en/latest/index.html)
* [UHD and USRP Manual (Ettus Research)](https://files.ettus.com/manual/page_install.html)
* [srsGUI Repository (GitHub)](https://github.com/srsran/srsgui)

## System Requirements

Ubuntu 20.04 LTS or later. 

## Installation from Source

1. Create working repository

    ```shell
   cd ~
   mkdir workspace
   cd workspace
   ```

2. Install UHD from source as RF front-end driver

   1. Set up dependencies
        
        ```shell
      sudo apt-get install autoconf automake build-essential ccache cmake cpufrequtils doxygen ethtool \ 
      g++ git inetutils-tools libboost-all-dev libncurses5 libncurses5-dev libusb-1.0-0 libusb-1.0-0-dev \ 
      libusb-dev python3-dev python3-mako python3-numpy python3-requests python3-scipy python3-setuptools \ 
      python3-ruamel.yaml 
      ```
      
   2. Download source code
   
        ```shell
      git clone https://github.com/EttusResearch/uhd.git
      ```

   3. Build and install
        
        ```shell
      cd uhd/host
      mkdir build
      cd build
      cmake ../
      make
      make test
      sudo make install
      sudo ldconfig
      ```
   
   4. (Optional) Replace default USRP hardware images by customized ones
        ```shell
       sudo mv /usr/local/share/uhd/images/usrp_b210_fpga.bin \ 
      /usr/local/share/uhd/images/usrp_b210_fpga_backup.bin
      sudo mv ~/Downloads/usrp_b210_fpga.bin /usr/local/share/uhd/images/usrp_b210_fpga.bin
      ```

   5. Verify installation by connecting to USRP devices
   
        ```shell
      sudo uhd_find_devices 
      ```
      
3. (Optional) Install srsGUI from real-time plotting

4. Install srsLTE from source

   1. Set up dependencies
   
       ```shell
      sudo apt-get install build-essential cmake libfftw3-dev libmbedtls-dev libboost-program-options-dev \
      libconfig++-dev libsctp-dev
       ```
   
   2. Download source code
    
        ```shell
       cd ~/Workspace
       git clone https://github.com/srsRAN/srsRAN_4G.git
      ```   

   3. Build and install
   
        ```shell
      cd srsRAN_4G
      mkdir build
      cd build
      cmake ../
      make
      make test
      sudo make install
      sudo ldconfig
      ```
   
   4. Set up configuration files to `/etc/.config/srsran`
   
        ```shell
      sudo srsran_install_config.sh service
      ```
