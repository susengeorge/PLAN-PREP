const bcrypt = require('bcryptjs');

bcrypt.hash("pwd", 10, (err, hashedPassword) => {
    if (err) console.error("Error:", err);
    else console.log("New Hashed Password:", hashedPassword);
});
