// ignore_for_file: prefer_const_constructors, deprecated_member_use, prefer_const_literals_to_create_immutables

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:fish/models/requestConsumption.model.dart';
import 'package:fish/models/responseConsumption.model.dart';
import 'package:fish/models/responseFilter.model.dart';
import 'package:fish/utils/const.dart';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class WaterColorScreen extends StatefulWidget {
  @override
  _WaterColorScreenState createState() => _WaterColorScreenState();
}

class _WaterColorScreenState extends State<WaterColorScreen> {
  late PageController _controller;
  int _selectedIndex = 0;
  bool _obscureText = true;

  String? _dropDownValue;
  String? _dropDownValueStage;

  final GlobalKey<ScaffoldState> _scaffoldKeyDemand =
      GlobalKey<ScaffoldState>();
  var dio = Dio();

  File? _image;
  final picker = ImagePicker();

  Image? _imageWidget;

  String? count;
  String phReason = "";
  TextEditingController _controllerName = TextEditingController();

  String? colorStatus;

  bool filterLoaded = false;
  @override
  void initState() {
    super.initState();

    _controller = PageController();
  }

  void _onTappedBar(int value) {
    setState(() {
      _selectedIndex = value;
    });
    _controller.jumpToPage(value);
  }

  Future getImage() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      _image = File(pickedFile!.path);
      _imageWidget = Image.file(_image!);
    });
  }

  Future getImageCamera() async {
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    setState(() {
      _image = File(pickedFile!.path);
      _imageWidget = Image.file(_image!);
    });
  }

  Future<void> _getcolorStatus(File file) async {
    String filePath = file.path.split('/').last;
    FormData formData = FormData.fromMap({
      "filePath": await MultipartFile.fromFile(file.path, filename: filePath),
    });
    Response response =
        await dio.post(BASE_URL + 'watercolor/', data: formData);

    colorStatus = ResponseFilter.fromJson(response.data).color;
    if (colorStatus != null) {
      setState(() {
        filterLoaded = true;
      });
    }
    print("filter res*********************");
    print(colorStatus);
  }

  void checkPhValue() {
    double phValue = double.parse(_controllerName.text);
    if (colorStatus == "DarkGreenWater") {
      if (1 <= phValue && phValue <= 6.7) {
        phReason =
            "Due to sudden population explosions of suspended algae (phytoplankton)";
      } else if (6.8 <= phValue && phValue <= 7.8) {
        phReason =
            "It is within the acceptable pH range along with the growth of algae";
      } else if (7.9 <= phValue && phValue <= 14) {
        phReason = "It encourages the growth of algae and slime";
      }
    } else if (colorStatus == "BrownWater") {
      if (1 <= phValue && phValue <= 6.7) {
        phReason =
            "Color change has occurred due to certain stones, rocks and carbonic suspended organic particles";
      } else if (6.8 <= phValue && phValue <= 7.8) {
        phReason =
            "It is within the acceptable pH range along with clay particles";
      } else if (7.9 <= phValue && phValue <= 10.9) {
        phReason =
            "Color change has occurred due to the existence of clay particles";
      } else if (11 <= phValue && phValue <= 14) {
        phReason =
            "Plants such as Driftwood, Peat moss and Catapppa Leaves release substances (Tannins)";
      }
    } else {
      if (1 <= phValue && phValue <= 2.4) {
        phReason = "Due to depletion of Carbondioxide";
      } else if (2.5 <= phValue && phValue <= 6.7) {
        phReason =
            "Parasite and Bacterial population in water is at a minimum level";
      } else if (6.8 <= phValue && phValue <= 7.8) {
        phReason = "Water is good within an acceptable pH range";
      } else if (7.9 <= phValue && phValue <= 10.9) {
        phReason =
            "Parasite and Bacterial population in water is at a maximum level";
      } else if (11 <= phValue && phValue <= 14) {
        phReason =
            "Colour change has occurred due to presence of certain stones";
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    double height = MediaQuery.of(context).size.height;
    double width = MediaQuery.of(context).size.width;
    return Scaffold(
        key: _scaffoldKeyDemand,
        resizeToAvoidBottomInset: false,
        bottomNavigationBar: BottomNavigationBar(
          onTap: _onTappedBar,
          items: [
            BottomNavigationBarItem(
                icon: Icon(Icons.takeout_dining), label: 'WaterColor'),
            BottomNavigationBarItem(icon: Icon(Icons.water), label: 'PhValue'),
          ],
          selectedItemColor: Colors.orange,
          currentIndex: _selectedIndex,
        ),
        appBar: AppBar(
          backgroundColor: Colors.blue,
          title: Text("Water Color"),
          leading: IconButton(
              onPressed: () {
                Navigator.of(context).pushReplacementNamed('/welcome');
              },
              icon: Icon(Icons.arrow_back_ios)),
        ),
        body: PageView(
          onPageChanged: (page) {
            setState(() {
              _selectedIndex = page;
            });
          },
          controller: _controller,
          children: <Widget>[
            ListView(
              children: [
                Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      _image == null
                          ? Text(
                              'No image selected.',
                              style: TextStyle(fontSize: 20),
                            )
                          : Container(
                              constraints: BoxConstraints(
                                  maxHeight:
                                      MediaQuery.of(context).size.height / 2),
                              decoration: BoxDecoration(
                                border: Border.all(),
                              ),
                              child: _imageWidget,
                            ),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 30, 0, 0),
                          child: RaisedButton(
                            onPressed: () => getImageCamera(),
                            child:
                                Text('Click Here To Select Image From Camera'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                          child: RaisedButton(
                            onPressed: () => getImage(),
                            child:
                                Text('Click Here To Select Image From Gallery'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                          child: RaisedButton(
                            onPressed: () => _getcolorStatus(_image!),
                            child: Text('CheckImage'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                    ]),
                SizedBox(height: 10),
                filterLoaded
                    ? Padding(
                        padding: const EdgeInsets.all(15.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              colorStatus! + " Tank Detected.",
                              style: TextStyle(fontSize: 20),
                            ),
                            SizedBox(
                              height: 10,
                            ),
                            colorStatus == "DarkGreenWater"
                                ? Column(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      Text(
                                        "Prediction",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "*.Due to sudden population explosions of suspended algae (phytoplankton)"),
                                      Text(
                                        "Description",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "Unlike other algae species that grow on the glass or objects in the aquarium, green water algae float about the tank and multiply by billions in a short period of time, in what's known as a bloom. Some algal blooms are the result of an excess of nutrients (particularly Phosphorus and Nitrogen) into waters, where higher concentrations of these nutrients in water cause increased growth of algae and green plants. When too many nutrients are available, phytoplankton may grow out of control and form harmful algal blooms (HABs)."),
                                      SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        "Prediction",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text("*.Due to excessive Light"),
                                      Text(
                                        "Description",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "Algae are plants, and plants love light. However, excessive light in the presence of a source of nutrients can send algae into hyperdrive. Too much light can come from placing your aquarium in a sunny window, leaving the tank light on too long, or using a light that is too strong for the aquarium."),
                                      SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        "Prediction",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "*.Due to nutrient imbalance in water"),
                                      Text(
                                        "Description",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "Plants need nutrients to grow. For most algae that means nitrate and phosphate, which typically come from fish food and fish waste, but they can be in tap water as well. Overfeeding and/or having too many fish for your tank size or filter capacity also lead to a build-up of nutrients. Performing water changes with nutrient-laden tap water will have the same effect."),
                                      SizedBox(
                                        height: 10,
                                      ),
                                      Text(
                                        "Prediction",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "*Color change has occurred due to poor maintenance habits"),
                                      Text(
                                        "Description",
                                        style: TextStyle(fontSize: 20),
                                      ),
                                      Text(
                                          "Lack of water changes and proper maintenance causes a gradual deterioration of water quality. Over time this creates an ideal environment for algae to grow. Since we can't see nutrients, we have no idea how bad conditions are, until something bad happens, like our fish getting sick or an algae bloom occurring."),
                                      SizedBox(
                                        height: 10,
                                      ),
                                    ],
                                  )
                                : colorStatus == "BrownWater"
                                    ? Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*.Tank Is Infected with Bacteria or Other Microorganisms"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "Bacteria could be turning the tank water cloudy and discolored. When the tank is unclean or exposed to bacteria, an overgrowth can form in the aquarium. This is commonly referred to as a bacteria bloom. When the tank is brown it is likely caused by a bloom of algae. Algae are micro-organisms that can taint the water and cause it to turn brown. Algae are usually green but when they sit in the water, they turn it into yellow mixed brown. The tank could be cloudy because of bacteria, and brown due to an abundance of waste and algae. "),
                                          SizedBox(
                                            height: 10,
                                          ),
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*.Due to the existence of Tannins in water"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "Natural objects like Driftwood or African leaves can actually release tannins in the tank water. Tannins are not harmful to an aquarium. They are a natural substance that discolors water, but it lowers the pH of a tank. In nature, tannins can discolor water and turn a fish habitat yellow or brown as well. The same thing happens in the tank with certain woods and plant material."),
                                          SizedBox(
                                            height: 10,
                                          ),
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*.Due to the existence of rotting organic material in your tank"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "Cloudy discolored water is a sign that the water has become heavily polluted. Organic materials like food, fish waste, plant waste, and other rotting substances can create a harmful environment for fish. A bad smell will usually accompany the brown tinge of the aquarium. This is caused when tanks are filled with rotting material or have a heavy bio load. Fish cannot survive long in this type of tank environment. They will get sick and die. Maybe the fish is producing a lot of waste and creates a concentrated and heavy bio load in the tank. If the tank doesn't have a filter, water will change colors. Alternatively, a lot of dead material can be floating in the tank and create problems. For instance, if the tank is a planted tank, stray leaves and foliage could detach from the plants and they turn the water yellow or brown."),
                                          SizedBox(
                                            height: 10,
                                          ),
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*.You have not washed the substrates well"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "New tanks are particularly prone to water problems. When the aquarium is set up, you might notice that your water is murky and yellow mixed brown. This could be due to the existence of unwashed substrates. The substrate has dust and potential contaminants on it. If you just put the substrate into your tank, water can turn into a different color. You want to try and wash your substrate before placing it into a new aquarium. This way your water doesn’t turn brown or cloudy. In some cases, certain substrates might have tannins in them as well. Your substrate might not be contaminated or harmful. But it’s still good to give it a rinse so you can keep your tank water clean and safe!"),
                                          SizedBox(
                                            height: 10,
                                          ),
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*.Color change has occurred due to the existence of clay particles"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "If there are clay particles in the water, the cloudiness is mostly likely caused by fish waste, excess food, dusty substrate, or other miscellaneous debris. When setting up a new tank or planting aquarium plants, tiny bits of substrate may float into the water column."),
                                          SizedBox(
                                            height: 10,
                                          ),
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*The carbonic suspended organic particles have increased in water such as fish fecal matter and carbonic garbage"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "When Carbonic substrates are dissolved highly in water, pH of the tank water increases as well. This affects negatively where Carbonic acid is formed, which results in problems caused by Ammonia"),
                                          SizedBox(
                                            height: 10,
                                          ),
                                        ],
                                      )
                                    : Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Text(
                                            "Prediction",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "*It provides a good environment for the survival of fish"),
                                          Text(
                                            "Description",
                                            style: TextStyle(fontSize: 20),
                                          ),
                                          Text(
                                              "Crystal clear, healthy-looking water is the aim of virtually every aquarium owner. If it is not properly maintained, aquarium water can quickly become cloudy with full of algae and get discoloured."),
                                          SizedBox(
                                            height: 10,
                                          )
                                        ],
                                      )
                          ],
                        ),
                      )
                    : Container(),
              ],
            ),
            Container(
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: filterLoaded
                    ? ListView(
                        children: [
                          TextFormField(
                            controller: _controllerName,
                            keyboardType: TextInputType.number,
                            decoration: InputDecoration(
                              labelText: 'Enter Ph value',
                            ),
                            onSaved: (val) {
                              count = val;
                            },
                            validator: (value) {
                              if (value!.isEmpty) {
                                return 'Please enter value';
                              }
                              return null;
                            },
                          ),
                          SizedBox(
                            height: 20,
                          ),
                          Center(
                            child: Container(
                                margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                                child: RaisedButton(
                                  onPressed: () => {checkPhValue()},
                                  child: Text('Enter'),
                                  textColor: Colors.white,
                                  color: Colors.blueAccent[100],
                                  padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                                )),
                          ),
                          SizedBox(
                            height: 15,
                          ),
                          Text(phReason)
                        ],
                      )
                    : Container(
                        child: Center(
                          child: Text(
                            "Check Image to Enter Ph Value",
                            style: TextStyle(fontSize: 25),
                          ),
                        ),
                      ),
              ),
            ),
          ],
          scrollDirection: Axis.horizontal,
          pageSnapping: true,
          physics: BouncingScrollPhysics(),
        ));
  }
}

class Species {
  const Species(this.id, this.name);

  final String name;
  final int id;
}

class FishStage {
  const FishStage(this.id, this.name);

  final String name;
  final int id;
}
