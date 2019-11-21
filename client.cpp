#include "socket.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <sys/types.h>

void clear_stdin(){
	int ch;
	while((ch==getchar()) !=EOF && ch != '\n'){};
}

int main(int argc, char* argv[])
{
    /*
    if(argc != 2)
    {    // argv[0]에 ./client가 들어간다.
    printf("사용법 : %s IPv4-address\n", argv[0]);
    return -1;
    }
    */
 	//std::system("OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES=1 ./1119 &\n");
    int client_socket;
    struct sockaddr_in serverAddress;
    unsigned int server_addr_size;
    char sendBuff[BUFF_SIZE];
    char readBuff[BUFF_SIZE];
    std::string run("run\n");
    std::string start("start\n");
 
 
 
    ssize_t receivedBytes;
    ssize_t sentBytes;
 
 
    memset(&serverAddress, 0, sizeof(serverAddress));
 
    serverAddress.sin_family = AF_INET;
    inet_aton("127.0.0.1", (struct in_addr*) &serverAddress.sin_addr.s_addr);
    serverAddress.sin_port = htons(20162);
 
    // 소켓 생성
    if ((client_socket = socket(PF_INET, SOCK_DGRAM, 0)) == -1)
    {
        printf("socket 생성 실패\n");
        exit(0);
    }
 
    while (1)
    {

        // 채팅 프로그램 제작
 
        server_addr_size = sizeof(serverAddress);
        //클라이언트에서 메세지 전송

         printf("클라이언트에서 보낼 말을 입력하세요 :: ");
        char msg[BUFF_SIZE];
	char *send_data;
        fgets(msg, BUFF_SIZE, stdin);
//int status;

        sprintf(sendBuff, "%s", msg);
	send_data=sendBuff;
//clear_stdin();
	if(run.compare(send_data)==0){
		//pid_t pid = fork();
		//if(!pid){
  			//printf("test-1");
std::system("sudo chmod a+rw /dev/ttyACM0");
std::system("nvidia");
			std::system(" nohup ./1119 OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES=1  &");
			std::system("\n");
  			//printf("test0");
        		//fflush(stdout);
 			//clear_stdin();
		//readBuff[3] = '\0';
		//fputs(readBuff, stdout);
        	//fflush(stdout);
		//	wait(&status);
		//}		

	}
	else if(start.compare(send_data)==0)
	{	
		std::system(" nohup ./rplidarnon &");
		//std::system(" ./rplidarnon");
		std::system("\n");
	}
	//else{
  	//printf("test1");
        sentBytes = sendto(client_socket, sendBuff, strlen(sendBuff), 0, (struct sockaddr*)&serverAddress, sizeof(serverAddress));
  	//printf("test2");
        //receivedBytes = recvfrom(client_socket, readBuff, BUFF_SIZE, 0, (struct sockaddr*)&serverAddress, &server_addr_size);
  	//printf("test3");
	//clear_stdin();
 	//printf("%lu bytes read\n", receivedBytes);
  	//printf("test4");
			
	readBuff[strlen(readBuff)-1] = '\0';
	fputs(readBuff, stdout);
        fflush(stdout);
	//clear_stdin();
	//}
       


    }
 
    // 소켓 close
    close(client_socket);
    return 0;
} 


