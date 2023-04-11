#include "qrw/Animator.hpp"
#include <termios.h>

namespace qrw {

class KeyboardInput : public AnimatorBase {
 public:
  KeyboardInput(Params const& params);
  ~KeyboardInput();

  static constexpr int FWD = 'z';
  static constexpr int BWD = 's';
  static constexpr int ESC = 27;
  enum ArrKeys { UP = 'A', DOWN = 'B', RIGHT = 'C', LEFT = 'D' };

  int listen();
  void keyhandler(int c) {}

  // listen to next two characters and decide
  void handlescaped();

  void update_v_ref(int k, bool gait_is_static) { int c = listen(); }

 private:
};

}  // namespace qrw
