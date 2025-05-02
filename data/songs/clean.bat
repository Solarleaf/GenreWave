@echo off
setlocal enabledelayedexpansion

:: Loop through all .mp3 files
for %%F in (*.mp3) do (
    set "name=%%~nF"
    set "ext=%%~xF"
    set "clean="

    :: Remove non-letter characters
    for /L %%i in (0,1,127) do (
        set "char=!name:~%%i,1!"
        for %%a in (A B C D E F G H I J K L M N O P Q R S T U V W X Y Z a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
            if "!char!"=="%%a" set "clean=!clean!!char!"
        )
    )

    if not "!clean!!ext!"=="%%F" (
        echo Renaming "%%F" to "!clean!!ext!"
        ren "%%F" "!clean!!ext!"
    )
)

echo Done.
pause
