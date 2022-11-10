// ignore_for_file: prefer_const_constructors, deprecated_member_use, prefer_const_literals_to_create_immutables

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:fish/models/requestConsumption.model.dart';
import 'package:fish/models/responseConsumption.model.dart';
import 'package:fish/models/responseFilter.model.dart';
import 'package:fish/utils/const.dart';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class FoodConsumption extends StatefulWidget {
  @override
  _FoodConsumptionState createState() => _FoodConsumptionState();
}

class _FoodConsumptionState extends State<FoodConsumption> {
  late PageController _controller;
  int _selectedIndex = 0;
  bool _obscureText = true;
  String? password = "sanju.ad";
  String? _dropDownValue;
  String? _dropDownValueStage;

  int? selectedSpecies;
  int? slectedFishStage;
  String? count;
  TextEditingController _controllerName = TextEditingController();

  double? foodAmount;
  double? foodAmountDay;
  double? foodAmountMonth;
  String? foodType;
  List<Species> species = [
    Species(0, "Acara"),
    Species(1, "Angel Fish"),
    Species(2, "Betta (Siamese Fighting Fish)"),
    Species(3, "Carp"),
    Species(4, "Flowerhorn Fish"),
    Species(5, "Gold Fish"),
    Species(6, "Guppy"),
    Species(7, "Neon Tetra"),
    Species(8, "Platy"),
    Species(9, "Rainbow Fish"),
    Species(10, "Swordtail"),
    Species(11, "Zebra Fish"),
    Species(12, "cara")
  ];

  List<FishStage> stages = [
    FishStage(0, "Breeding"),
    FishStage(1, "Growing"),
    FishStage(2, "Nursery")
  ];

  final GlobalKey<ScaffoldState> _scaffoldKeyDemand =
      GlobalKey<ScaffoldState>();
  var dio = Dio();

  File? _image;
  final picker = ImagePicker();

  Image? _imageWidget;

  String? filterStatus;
  String? filterIns;
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

  Future<void> _getFilterStatus(File file) async {
    String filePath = file.path.split('/').last;
    FormData formData = FormData.fromMap({
      "filePath": await MultipartFile.fromFile(file.path, filename: filePath),
    });
    Response response = await dio.post(BASE_URL + 'filter/', data: formData);

    filterStatus = ResponseFilter.fromJson(response.data).color;
    if (filterStatus != null) {
      setState(() {
        filterLoaded = true;
      });

      if (filterStatus == "GoodConditionFilter") {
        setState(() {
          filterIns = "Filter can be reused.No need to replace";
        });
      } else {
        setState(() {
          filterIns = "Filter can't be reused.Need to replace immediately";
        });
      }
    }
    print("filter res*********************");
    print(filterStatus);
  }

  Future<void> _getFoodConsumption(
      int fishCount, int species, int fishStage) async {
    print("**********_getFoodConsumption called");
    RequestConsumption request = RequestConsumption(
        fishCount: fishCount, species: species, fishStage: fishStage);
    Response response =
        await dio.post(BASE_URL + 'consumption/', data: request.toJson());

    setState(() {
      foodAmount = ResponseConsumption.fromJson(response.data).foodAmount;
      foodType = ResponseConsumption.fromJson(response.data).foodType;
      foodAmountDay = foodAmount! * 3;
      foodAmountMonth = foodAmountDay! * 30;
    });

    print("Food type iss*********");
    print(foodType);
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
                icon: Icon(Icons.takeout_dining), label: 'Filter'),
            BottomNavigationBarItem(
                icon: Icon(Icons.water), label: 'Consumption'),
          ],
          selectedItemColor: Colors.orange,
          currentIndex: _selectedIndex,
        ),
        appBar: AppBar(
          backgroundColor: Colors.blue,
          title: Text("Consumption & Filter"),
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
                            onPressed: () => _getFilterStatus(_image!),
                            child: Text('CheckFilter'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      SizedBox(height: 30),
                      filterLoaded
                          ? Column(
                              children: [
                                Text(filterStatus!),
                                SizedBox(
                                  height: 10,
                                ),
                                Text(filterIns!),
                              ],
                            )
                          : Container()
                    ])
              ],
            ),
            Container(
              child: Padding(
                padding: const EdgeInsets.all(20.0),
                child: ListView(
                  children: [
                    Text("Select the Species"),
                    DropdownButton<Species>(
                      hint: _dropDownValue == null
                          ? Text('SelectSpecies')
                          : Text(
                              _dropDownValue!,
                              style: TextStyle(color: Colors.black),
                            ),
                      isExpanded: true,
                      iconSize: 30.0,
                      style: TextStyle(color: Colors.black),
                      items: species.map((Species species) {
                        return DropdownMenuItem<Species>(
                          value: species,
                          child: Text(
                            species.name,
                            style: TextStyle(color: Colors.black),
                          ),
                        );
                      }).toList(),
                      onChanged: (val) {
                        setState(
                          () {
                            selectedSpecies = val!.id;
                            _dropDownValue = val.name;
                            // _getDemand(selectedSpecies!);
                          },
                        );
                      },
                    ),
                    SizedBox(
                      height: 20,
                    ),
                    Text("Select the Fish Stage"),
                    DropdownButton<FishStage>(
                      hint: _dropDownValueStage == null
                          ? Text('SelectSpecies')
                          : Text(
                              _dropDownValueStage!,
                              style: TextStyle(color: Colors.black),
                            ),
                      isExpanded: true,
                      iconSize: 30.0,
                      style: TextStyle(color: Colors.black),
                      items: stages.map((FishStage satges) {
                        return DropdownMenuItem<FishStage>(
                          value: satges,
                          child: Text(
                            satges.name,
                            style: TextStyle(color: Colors.black),
                          ),
                        );
                      }).toList(),
                      onChanged: (val) {
                        setState(
                          () {
                            slectedFishStage = val!.id;
                            _dropDownValueStage = val.name;
                          },
                        );
                      },
                    ),
                    SizedBox(
                      height: 20,
                    ),
                    Text("Enter the Fish Count"),
                    TextFormField(
                      controller: _controllerName,
                      decoration: InputDecoration(
                        labelText: 'Fish Count',
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
                            onPressed: () => {
                              _getFoodConsumption(
                                  int.parse(_controllerName.text),
                                  selectedSpecies!,
                                  slectedFishStage!),
                            },
                            child: Text('Enter'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                    ),
                    SizedBox(
                      height: 20,
                    ),
                    foodAmount != null
                        ? Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text("Predicted Food Amount is: " +
                                  foodAmount!.toStringAsFixed(2)),
                              SizedBox(
                                height: 10,
                              ),
                              Text("Per meal: " +
                                  foodAmount!.toStringAsFixed(2)),
                              SizedBox(
                                height: 10,
                              ),
                              Text("Per day: " +
                                  foodAmountDay!.toStringAsFixed(2)),
                              SizedBox(
                                height: 10,
                              ),
                              Text("Per month: " +
                                  foodAmountMonth!.toStringAsFixed(2)),
                            ],
                          )
                        : Container(),
                    SizedBox(
                      height: 20,
                    ),
                    foodType != null
                        ? Text("Predicted Food Type is: " + foodType.toString())
                        : Container(),
                  ],
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
