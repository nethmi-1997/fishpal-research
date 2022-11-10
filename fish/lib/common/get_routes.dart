import 'package:fish/screens/behaviour.screen.dart';
import 'package:fish/screens/fish_count.screen.dart';
import 'package:fish/screens/food_consumption.screen.dart';
import 'package:fish/screens/splash.dart';
import 'package:fish/screens/water_color.screen.dart';
import 'package:fish/screens/welcome.screen.dart';
import 'package:flutter/material.dart';

class Navigate {
  static Map<String, Widget Function(BuildContext)> routes = {
    // '/': (context) => WelcomePage(),
    '/splash': (context) => SplashScreen(),
    '/welcome': (context) => WelcomeScreen(),
    '/foodConsumption': (context) => FoodConsumption(),
    '/waterColor': (context) => WaterColorScreen(),
    '/fishCount': (context) => FishCountScreen(),
    '/behaviour': (context) => BehaviourScreen(),
  };
}
