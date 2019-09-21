# Boston Housing Prediction
> Predict Housing prices in boston with different Models.   
  
[![License][license-badge]][license-url]
[![Codacy Badge][codacy-badge]][codacy-url]
[![Build Status][travis-badge]][travis-url]
![Lines of Code][lines-codes-badge]  
Boston Housing Prediction is a python script that can predict the housing prices in boston with different models, the user can choose from.  

![header](res/img/script_preview_scaled.gif)

## Installation
You need to have `python >= 3.5` installed.

To install the `requirements` for the script do:  

```sh
$ pip install -r requirements.txt
```

Download the latest release of the script `boston-housing-X_X_X.py` You can also download the script under release.

## Usage example

You can run the programm with(The X stand for the version):
```sh
$ py boston-housing-X_X_X.py linear_regression
```  
To see the help(for extra options) do:
```sh
$ py boston-housing-X_X_X.py -h
```  

<!--_For more examples and usage, please refer to the [Wiki][wiki]._-->

## Release History

*   0.1.2
    *   bugfixes #4, #5
    *   serperated code in diffeent files for more clarity. See code folder

*   0.1.1
    *   added functionality to load models without training
    *   plots are now outsourced and handled by a different kernel
    *   dataset gets automatically downloaded when missing
    *   v_data shows 4 different plots to describe the data(previously 2).
    
*   0.1.0  
    *   The first proper release
    *   Realese of readme (Thanks @dbader)
       
*   0.0.1  
    *   Work in progress

## Roadmap (planned updates)

*   Add more models

    *   polynomial regression
    *   normal equation
    *   svm
    *   neural network

*   Upload pre-trained models 

## Meta

<!--Your Name – [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com-->

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/LuposX](https://github.com/LuposX)

## Contributing

1.  Fork it (<https://github.com/LuposX/BostonHousingPrediction/fork>)
2.  Create your feature branch (`git checkout -b feature/fooBar`)
3.  Commit your changes (`git commit -am 'Add some fooBar'`)
4.  Push to the branch (`git push origin feature/fooBar`)
5.  Create a new Pull Request

<!-- Markdown link & img dfn's -->
[codacy-badge]: https://api.codacy.com/project/badge/Grade/089e59afa6a44e629b1267f8abaad038
[codacy-url]:https://app.codacy.com/manual/LuposX/BostonHousingPrediction/dashboard
[license-badge]: https://img.shields.io/github/license/LuposX/sentdex_fixed_market_stock
[license-url]: https://github.com/LuposX/BostonHousingPrediction/blob/master/LICENSE
[travis-url]: https://travis-ci.com/LuposX/BostonHousingPrediction
[travis-badge]: https://travis-ci.com/LuposX/BostonHousingPrediction.svg?branch=master
[lines-codes-badge]: https://tokei.rs/b1/github/LuposX/BostonHousingPrediction?category=code
