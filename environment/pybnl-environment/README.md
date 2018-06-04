
This subdirectory can be used to verify the build via docker without constantly committing your changes to github

Suppose you're in your `work` directory and you have checked out `pybnl` in that directory then:

    cp -r pybnl/environment/pybnl-environment .
    cd pybnl-environment
    make
