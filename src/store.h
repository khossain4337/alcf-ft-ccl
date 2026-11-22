// Key-value store as files in one dir on /vast. A TCP store can replace this file (DECISIONS_2026_11_22a).
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

struct Store {
  std::string dir;

  void put(const std::string& key, const std::string& val) const {
    std::string path = dir + "/" + key, tmp = path + ".tmp";  // rename is atomic: no partial reads
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f || fwrite(val.data(), 1, val.size(), f) != val.size() || fclose(f) || rename(tmp.c_str(), path.c_str())) {
      perror(tmp.c_str());
      exit(1);
    }
  }

  // readdir, not a name lookup: NFS caches "not found" for ~30 s on other nodes.
  bool wait(const std::string& key, double timeoutS) const {
    auto t0 = std::chrono::steady_clock::now();
    for (;;) {
      DIR* d = opendir(dir.c_str());
      bool found = false;
      for (dirent* e; d && !found && (e = readdir(d));) found = key == e->d_name;
      if (d) closedir(d);
      if (found) return true;
      if (std::chrono::steady_clock::now() - t0 > std::chrono::duration<double>(timeoutS)) return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  std::string get(const std::string& key) const {
    std::ifstream f(dir + "/" + key, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
  }
};
