#include <math.h>
#include <stdio.h>

double jawn() {
  printf("we're here now");

  return pow(3, 2);
}

double sig(double z) {
  return 1 / (1 + exp(-z));
}

int main() {
  printf("%f", jawn());
  
  return 0;
}
