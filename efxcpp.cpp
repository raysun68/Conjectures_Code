// efx_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <algorithm>

namespace py = pybind11;
using namespace std;

using Utilities = vector<vector<int>>;
using Allocation = vector<vector<int>>;

int bundle_value(const Utilities& utils, int agent, const vector<int>& bundle) {
  int value = 0;
  for (int item : bundle)
    value += utils[agent][item];
  return value;
}

bool is_efx(const Utilities& utils, const Allocation& allocation) {
  int n = utils.size();
  for (int i = 0; i < n; i++) {
    int i_value = bundle_value(utils, i, allocation[i]);
    for (int j = 0; j < n; j++) {
      if (i == j || allocation[j].empty()) continue;
      for (int item : allocation[j]) {
        vector<int> j_minus = allocation[j];
        j_minus.erase(remove(j_minus.begin(), j_minus.end(), item), j_minus.end());
        if (bundle_value(utils, i, j_minus) > i_value)
          return false;
      }
    }
  }
  return true;
}

PYBIND11_MODULE(efx, m) {
  m.def("is_efx", &is_efx, "Check if an allocation is EFX");
}