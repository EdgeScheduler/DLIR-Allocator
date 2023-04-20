#include <nlohmann/json.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <string.h>
#include <memory>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <sstream>
#include "RPCResponse/IPs.h"

std::string GetLocalIPs()
{
    nlohmann::json ips;
    //std::vector<std::string> ips;
	#define MAXINTERFACES 16 
    char pszIPBuf[16] = {0};
	bool result = false;
 
	int fd = -1;
	int interface = 0; 
	struct ifreq buf[MAXINTERFACES]; 
	struct ifconf ifc; 
 
	if ((fd = socket (AF_INET, SOCK_DGRAM, 0)) >= 0) 
	{ 
		ifc.ifc_len = sizeof(buf); 
		ifc.ifc_buf = (caddr_t) buf; 
		if (!ioctl (fd, SIOCGIFCONF, (char *) &ifc)) 
		{ 
			//获取接口信息
			interface = ifc.ifc_len / sizeof (struct ifreq); 
			//根据借口信息循环获取设备IP和MAC地址
			while (interface>=0) 
			{    
                //获取当前网卡的IP地址 
                if (!(ioctl (fd, SIOCGIFADDR, (char *) &buf[interface]))) 
                { 
                    if ( inet_ntop(AF_INET,&(( (struct sockaddr_in*)(& (buf[interface].ifr_addr) ))->sin_addr),pszIPBuf,16) )
                    {
                        ips[std::string(buf[interface].ifr_name)]=std::string(pszIPBuf);
                        memset(pszIPBuf,0,16);
                    }
                }
                interface--;
			}
            
		}
 
		close (fd); 
	}
 
	std::stringstream sstream;
	sstream << ips;
	std::string json_str;
	sstream>>json_str;
	return json_str;
}