#include "interface.h"
#include <iostream>

int main() {
  std::cout << "Which are you?" << std::endl;
  for (int64_t i = 0; i < 10; i++) {
    std::cout << "Learn " << i + 1 << std::endl;
    Learn(i);
  }
}