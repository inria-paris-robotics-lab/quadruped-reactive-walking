#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

/// @brief Expand environment variables
/// @param text the text string containing environment variables
inline std::string expand_env(std::string text)
{
  static const std::regex env_re{R"--(\$\{([^}]+)\})--"};
  std::smatch match;
  while (std::regex_search(text, match, env_re))
  {
    auto const from = match[0];
    auto const var_name = match[1].str();
    text.replace(from.first, from.second, std::getenv(var_name.c_str()));
  }
  return text;
}

#endif // UTILS_H_INCLUDED
