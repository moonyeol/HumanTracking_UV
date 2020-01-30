#include "socket.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <sys/types.h>

void clear_stdin() {
	int ch;
	while ((ch == getchar()) != EOF && ch != '\n') {};
}

int main(int argc, char* argv[])
{
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
	inet_aton("127.0.0.1", (struct in_addr*) & serverAddress.sin_addr.s_addr);
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
		char* send_data;
		fgets(msg, BUFF_SIZE, stdin);
		sprintf(sendBuff, "%s", msg);
		send_data = sendBuff;
		if (run.compare(send_data) == 0) {
			std::system("sudo chmod a+rw /dev/ttyTHS0");
			std::system("nvidia");
			std::system(" nohup ./main OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES=1  &");
			std::system("\n");
		}
		else if (start.compare(send_data) == 0)
		{
			std::system(" nohup ./rplidar_revision.out &");
			std::system("\n");
		}
		sentBytes = sendto(client_socket, sendBuff, strlen(sendBuff), 0, (struct sockaddr*) & serverAddress, sizeof(serverAddress));
		readBuff[strlen(readBuff) - 1] = '\0';
		fputs(readBuff, stdout);
		fflush(stdout);
	}
	// 소켓 close
	close(client_socket);
	return 0;
}
