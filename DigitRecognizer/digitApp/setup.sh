mkdir -p ~/.digitApp/

echo "\
[general]\n\
email = \"ibrahim.karmadzha@gmail.com\"\n\
" > ~/.digitApp/credentials.toml

echo "\
[server]\n\
headless = true\n\
emableCORS = false\n\
port = $PORT\n\
" > ~/.digitApp/config.toml