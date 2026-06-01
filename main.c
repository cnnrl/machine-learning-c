#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

struct model { // y = w_1 * x_1 + w_2 * x_2 + b
  double w_1;
  double w_2;
  double w_3;
  double w_4;
  double v_1;
  double v_2;
  double b_1;
  double b_2;
  double b_3;
};

int training_data[][3] = {
    {0, 0, 0},
    {1, 0, 1},
    {0, 1, 1},
    {1, 1, 0},
};

int SIZE = 4;

double relu(double z) { return (z > 0) ? z : 0.01 * z; }
double sig(double z) { return 1.0 / (1.0 + exp(-z)); }

double cost(double y_0, double y) { // (y_0 -y)^2
  return pow(y - y_0, 2);
}

double eval(double n1, double n2, double b, double x1, double x2) {
  return n1 * x1 + n2 * x2 + b;
}

int main() {
  struct model m;
  srand(time(NULL));
  m.w_1 = (double)rand() / RAND_MAX;
  m.w_2 = (double)rand() / RAND_MAX;
  m.w_3 = (double)rand() / RAND_MAX;
  m.w_4 = (double)rand() / RAND_MAX;
  m.b_1 = (double)rand() / RAND_MAX;
  m.b_2 = (double)rand() / RAND_MAX;
  m.b_3 = (double)rand() / RAND_MAX;
  m.v_1 = (double)rand() / RAND_MAX;
  m.v_2 = (double)rand() / RAND_MAX;
  double h = 1e-3;
  double eta = 1e-1;

  for (int i = 0; i < 100 * 100; ++i) {
    for (int j = 0; j < SIZE; ++j) {
      int x_1 = training_data[j][0];
      int x_2 = training_data[j][1];
      int y = training_data[j][2];
      double h_1 = m.w_1 * x_1 + m.w_2 * x_2 + m.b_1;
      double z_1 = relu(h_1);
      double h_2 = m.w_3 * x_1 + m.w_4 * x_2 + m.b_2;
      double z_2 = relu(h_2);
      double h_3 = m.v_1 * z_1 + m.v_2 * z_2 + m.b_3;
      double y_0 = relu(h_3);
      double c = cost(y_0, y);

      double dw_1 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3,
                          relu(eval(m.w_1 + h, m.w_2, m.b_1, x_1, x_2)), z_2)),
                y) -
           c) /
          h;
      double dw_2 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3,
                          relu(eval(m.w_1, m.w_2 + h, m.b_1, x_1, x_2)), z_2)),
                y) -
           c) /
          h;
      double db_1 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3,
                          relu(eval(m.w_1, m.w_2, m.b_1 + h, x_1, x_2)), z_2)),
                y) -
           c) /
          h;

      double dw_3 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3, z_1,
                          relu(eval(m.w_3 + h, m.w_4, m.b_2, x_1, x_2)))),
                y) -
           c) /
          h;
      double dw_4 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3, z_1,
                          relu(eval(m.w_3, m.w_4 + h, m.b_2, x_1, x_2)))),
                y) -
           c) /
          h;
      double db_2 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3, z_1,
                          relu(eval(m.w_3, m.w_4, m.b_2 + h, x_1, x_2)))),
                y) -
           c) /
          h;

      double dv_1 =
          (cost(relu(eval(m.v_1 + h, m.v_2, m.b_3, z_1, z_2)), y) - c) / h;
      double dv_2 =
          (cost(relu(eval(m.v_1, m.v_2 + h, m.b_3, z_1, z_2)), y) - c) / h;
      double db_3 =
          (cost(relu(eval(m.v_1, m.v_2, m.b_3 + h, z_1, z_2)), y) - c) / h;

      m.w_1 -= eta * dw_1;
      m.w_2 -= eta * dw_2;
      m.b_1 -= eta * db_1;

      m.w_3 -= eta * dw_3;
      m.w_4 -= eta * dw_4;
      m.b_2 -= eta * db_2;

      m.v_1 -= eta * dv_1;
      m.v_2 -= eta * dv_2;
      m.b_3 -= eta * db_3;
    }
  }

  printf("\n--- Final Model Predictions ---\n");
  for (int j = 0; j < SIZE; ++j) {
    double z_1 =
        relu(m.w_1 * training_data[j][0] + m.w_2 * training_data[j][1] + m.b_1);
    double z_2 =
        relu(m.w_3 * training_data[j][0] + m.w_4 * training_data[j][1] + m.b_2);
    double y_0 = relu(m.v_1 * z_1 + m.v_2 * z_2 + m.b_3);
    printf("Input: %d, %d | Target: %d | Predicted: %f\n", training_data[j][0],
           training_data[j][1], training_data[j][2], y_0);
  }
  return 0;
}
