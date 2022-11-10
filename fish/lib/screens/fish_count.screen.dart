// ignore_for_file: prefer_const_constructors, deprecated_member_use, prefer_const_literals_to_create_immutables

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:fish/models/requestConsumption.model.dart';
import 'package:fish/models/responseConsumption.model.dart';
import 'package:fish/models/responseFilter.model.dart';
import 'package:fish/models/responseObjecDetection.model.dart';
import 'package:fish/utils/const.dart';
import 'package:flutter/cupertino.dart';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class FishCountScreen extends StatefulWidget {
  @override
  _FishCountScreenState createState() => _FishCountScreenState();
}

class _FishCountScreenState extends State<FishCountScreen> {
  late PageController _controller;
  int _selectedIndex = 0;

  final GlobalKey<ScaffoldState> _scaffoldKeyDemand =
      GlobalKey<ScaffoldState>();
  var dio = Dio();

  final picker = ImagePicker();

  File? _image;
  Image? _imageWidget;

  File? _imageSide;
  Image? _imageWidgetSide;

  File? _imageLength;
  Image? _imageWidgetLength;

  String? count;

  double? fishCountTop;
  double? fishCountSide;
  double? fishLength;

  bool topLoaded = false;
  bool sideLoaded = false;
  bool lengthLoaded = false;
  bool showLoadingIndicator = false;
  @override
  void initState() {
    super.initState();
    dio.interceptors.add(LogInterceptor(responseBody: false));
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

  Future getImageSide() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      _imageSide = File(pickedFile!.path);
      _imageWidgetSide = Image.file(_imageSide!);
    });
  }

  Future getImageCameraSide() async {
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    setState(() {
      _imageSide = File(pickedFile!.path);
      _imageWidgetSide = Image.file(_imageSide!);
    });
  }

  Future getImageLength() async {
    final pickedFile = await picker.getImage(source: ImageSource.gallery);

    setState(() {
      _imageLength = File(pickedFile!.path);
      _imageWidgetLength = Image.file(_imageLength!);
    });
  }

  Future getImageCameraLength() async {
    final pickedFile = await picker.getImage(source: ImageSource.camera);

    setState(() {
      _image = File(pickedFile!.path);
      _imageWidget = Image.file(_image!);
    });
  }

  Future<void> _getTopCount(File file) async {
    String filePath = file.path.split('/').last;
    FormData formData = FormData.fromMap({
      "filePath": await MultipartFile.fromFile(file.path, filename: filePath),
    });
    setState(() {
      showLoadingIndicator = true;
    });
    Response response = await dio.post(BASE_URL + 'topFish/', data: formData);

    fishCountTop = ResponseObjectDetection.fromJson(response.data).objectRes;
    if (fishCountTop != null) {
      setState(() {
        topLoaded = true;
        showLoadingIndicator = false;
      });
    }
    print("top count*********************");
    print(fishCountTop);
  }

  Future<void> _getSideCount(File file) async {
    String filePath = file.path.split('/').last;
    FormData formData = FormData.fromMap({
      "filePath": await MultipartFile.fromFile(file.path, filename: filePath),
    });
    setState(() {
      showLoadingIndicator = true;
    });
    Response response = await dio.post(BASE_URL + 'sideFish/', data: formData);

    fishCountSide = ResponseObjectDetection.fromJson(response.data).objectRes;
    if (fishCountSide != null) {
      setState(() {
        sideLoaded = true;
        showLoadingIndicator = false;
      });
    }
    print("top count*********************");
    print(fishCountSide);
  }

  Future<void> _getLength(File file) async {
    String filePath = file.path.split('/').last;
    FormData formData = FormData.fromMap({
      "filePath": await MultipartFile.fromFile(file.path, filename: filePath),
    });
    setState(() {
      showLoadingIndicator = true;
    });
    Response response =
        await dio.post(BASE_URL + 'fishLength/', data: formData);

    fishLength = ResponseObjectDetection.fromJson(response.data).objectRes;
    if (fishLength != null) {
      setState(() {
        lengthLoaded = true;
        showLoadingIndicator = false;
      });
    }
    print("length******************");
    print(fishLength);
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
                icon: Icon(Icons.takeout_dining), label: 'TopView'),
            BottomNavigationBarItem(
                icon: Icon(Icons.trip_origin_outlined), label: 'SideView'),
            BottomNavigationBarItem(
                icon: Icon(Icons.legend_toggle), label: 'FishLength'),
          ],
          selectedItemColor: Colors.orange,
          currentIndex: _selectedIndex,
        ),
        appBar: AppBar(
          backgroundColor: Colors.blue,
          title: Text("Fish Count & Length"),
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
                            onPressed: () => _getTopCount(_image!),
                            child: Text('GetCount'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                    ]),
                topLoaded
                    ? Center(
                        child: Text(
                          "Fish count is:" + fishCountTop!.toString(),
                          style: TextStyle(fontSize: 20),
                        ),
                      )
                    : Container(),
                if (showLoadingIndicator)
                  Padding(
                      padding: EdgeInsets.only(
                        top: 18,
                      ),
                      child: SizedBox(child: CupertinoActivityIndicator()))
              ],
            ),
            ListView(
              children: [
                Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      _imageSide == null
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
                              child: _imageWidgetSide,
                            ),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 30, 0, 0),
                          child: RaisedButton(
                            onPressed: () => getImageCameraSide(),
                            child:
                                Text('Click Here To Select Image From Camera'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                          child: RaisedButton(
                            onPressed: () => getImageSide(),
                            child:
                                Text('Click Here To Select Image From Gallery'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                          child: RaisedButton(
                            onPressed: () => _getSideCount(_imageSide!),
                            child: Text('GetSideCount'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                    ]),
                sideLoaded
                    ? Center(
                        child: Text(
                          "Fish count is:" + fishCountSide!.toString(),
                          style: TextStyle(fontSize: 20),
                        ),
                      )
                    : Container(),
                if (showLoadingIndicator)
                  Padding(
                      padding: EdgeInsets.only(
                        top: 18,
                      ),
                      child: SizedBox(child: CupertinoActivityIndicator()))
              ],
            ),
            ListView(
              children: [
                Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: <Widget>[
                      _imageLength == null
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
                              child: _imageWidgetLength,
                            ),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 30, 0, 0),
                          child: RaisedButton(
                            onPressed: () => getImageCameraLength(),
                            child:
                                Text('Click Here To Select Image From Camera'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                          child: RaisedButton(
                            onPressed: () => getImageLength(),
                            child:
                                Text('Click Here To Select Image From Gallery'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                      Container(
                          margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                          child: RaisedButton(
                            onPressed: () => _getLength(_imageLength!),
                            child: Text('GetLength'),
                            textColor: Colors.white,
                            color: Colors.blueAccent[100],
                            padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                          )),
                    ]),
                lengthLoaded
                    ? Center(
                        child: Text(
                            "Fish Length is" + fishLength!.toString() + " cm"))
                    : Container(),
                if (showLoadingIndicator)
                  Padding(
                      padding: EdgeInsets.only(
                        top: 18,
                      ),
                      child: SizedBox(child: CupertinoActivityIndicator()))
              ],
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
