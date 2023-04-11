#include "qrw/Keyboard.hpp"
#include <unistd.h>
#include <ctype.h>

#define CTRL_KEY(k) ((k)&0x1f)

namespace qrw {

static struct termios oldt;

void resetTerminal() {}

void configTerminal() {
  struct termios newt;
  // get current term attributes
  tcgetattr(STDIN_FILENO, &newt);

  newt.c_lflag &= ~(ECHO | ICANON);

  tcsetattr(STDIN_FILENO, TCSAFLUSH, &newt);
}

void KeyboardInput::handlescaped() {
  int c[2];
  c[0] = getchar();
  c[1] = getchar();

  switch (c[0]) {
    case '[':
      printf("> got escape sequence.\n");
      switch (c[1]) {
        case UP:
        case DOWN:
        case RIGHT:
        case LEFT:
          break;
      }
    default:
      break;
  }
}

KeyboardInput::KeyboardInput(Params const& params) : AnimatorBase(params) {
  printf("Initialized keyboard input handler.\n");
  tcgetattr(STDIN_FILENO, &oldt);
  configTerminal();
}

void debugPrint(char c) {
  if (iscntrl(c)) {
    printf("char: %d\n", c);
  } else {
    printf("char: %d ('%c')\n", c, c);
  }
}

int KeyboardInput::listen() {
  // consure char
  char c;
  while (read(STDIN_FILENO, &c, sizeof(c)) == 1) {
    debugPrint(c);
    switch (c) {
      case ESC:
        handlescaped();
        break;
      case 'z':
        printf("forward\n");
        break;
      case 's':
        printf("backward\n");
        break;
      case -1:
      default:
        return c;
    }
    return c;
  }
  printf("done listening\n");

  return c;
}

KeyboardInput::~KeyboardInput() {
  printf("Ending keyboard handler.\n");
  resetTerminal();
}

}  // namespace qrw
