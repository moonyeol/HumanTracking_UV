#include <iostream>
class direction
{
private:
	char* move;
	long xcenter, tempsize = 0, size;

public:
	char* compare(long left, long right)
	{
		size = right - left;
		xcenter = (right + left) / 2;

		if (tempsize = 0)
		{
			tempsize = size;
		}
		else if (xcenter<100)
		{
			move = "l";
		}
		else if (xcenter > 520)
		{
			move = "r";
		}
		else if (tempsize < size - 5)
		{
			tempsize = size;
			move = "b";
		}
		else if (tempsize > size + 5)
		{
			tempsize = size;
			move = "g";
		}
		else
		{
			move = "s";
		}

		return move;
	}
};
using namespace std;
int main(void)
{
	char* move;
	direction direction;
	move = direction.compare(100, 200);
	cout << move << endl;
	return 0;
	
	
}
