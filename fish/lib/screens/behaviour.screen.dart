// ignore_for_file: prefer_const_constructors, deprecated_member_use, prefer_const_literals_to_create_immutables

import 'dart:io';

import 'package:dio/dio.dart';
import 'package:fish/models/requestConsumption.model.dart';
import 'package:fish/models/responseBehaviour.model.dart';
import 'package:fish/models/responseConsumption.model.dart';
import 'package:fish/models/responseFilter.model.dart';
import 'package:fish/models/responseObjecDetection.model.dart';
import 'package:fish/utils/const.dart';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:video_player/video_player.dart';

class BehaviourScreen extends StatefulWidget {
  @override
  _BehaviourScreenState createState() => _BehaviourScreenState();
}

class _BehaviourScreenState extends State<BehaviourScreen> {
  late PageController _controller;
  int _selectedIndex = 0;

  final GlobalKey<ScaffoldState> _scaffoldKeyDemand =
      GlobalKey<ScaffoldState>();
  var dio = Dio();

  File? _video;
  final picker = ImagePicker();

  VideoPlayerController? _controllers;
  Future<void>? _videos;

  Image? _imageWidget;

  String? behaviourType;

  bool filterLoaded = true;
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
    final pickedFile = await picker.pickVideo(source: ImageSource.gallery);

    setState(() {
      _video = File(pickedFile!.path);
      _controllers = VideoPlayerController.file(_video!);
      _videos = _controllers!.initialize();
    });
  }

  Future getImageCamera() async {
    final pickedFile = await picker.pickVideo(source: ImageSource.camera);

    setState(() {
      _video = File(pickedFile!.path);
      _controllers = VideoPlayerController.file(_video!);
      _videos = _controllers!.initialize();
    });
  }

  Future<void> _getbehaviourType(File file) async {
    setState(() {
      filterLoaded = false;
    });
    String filePath = file.path.split('/').last;
    FormData formData = FormData.fromMap({
      "filePath": await MultipartFile.fromFile(file.path, filename: filePath),
    });
    Response response = await dio.post(BASE_URL + 'behaviour/', data: formData);

    behaviourType = ResponseBehaviour.fromJson(response.data).type;
    if (behaviourType != null) {
      setState(() {
        filterLoaded = true;
      });
    }
    print("filter res*********************");
    print(behaviourType);
  }

  @override
  Widget build(BuildContext context) {
    double height = MediaQuery.of(context).size.height;
    double width = MediaQuery.of(context).size.width;
    return Scaffold(
        key: _scaffoldKeyDemand,
        resizeToAvoidBottomInset: false,
        floatingActionButton: FloatingActionButton(
          child: Icon(Icons.play_arrow),
          onPressed: () {
            if (_controllers!.value.isPlaying) {
              setState(() {
                _controllers!.pause();
              });
            } else {
              setState(() {
                _controllers!.play();
              });
            }
          },
          // child: Icon(
          //     _controllers!.value.isPlaying ? Icons.pause : Icons.play_arrow),
        ),
        appBar: AppBar(
          backgroundColor: Colors.blue,
          title: Text("Behaviour Detector"),
          leading: IconButton(
              onPressed: () {
                Navigator.of(context).pushReplacementNamed('/welcome');
              },
              icon: Icon(Icons.arrow_back_ios)),
        ),
        body: ListView(children: [
          Column(mainAxisAlignment: MainAxisAlignment.center, children: <
              Widget>[
            _video == null
                ? Text(
                    'No image selected.',
                    style: TextStyle(fontSize: 20),
                  )
                : FutureBuilder(
                    future: _videos,
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.done) {
                        return Center(
                          child: SizedBox(
                            height: 300,
                            child: VideoPlayer(_controllers!),
                          ),
                        );
                      } else {
                        return Center(
                          child: CircularProgressIndicator(),
                        );
                      }
                    },
                  ),
            Container(
                margin: EdgeInsets.fromLTRB(0, 30, 0, 0),
                child: RaisedButton(
                  onPressed: () => getImageCamera(),
                  child: Text('Click Here To Select Video From Camera'),
                  textColor: Colors.white,
                  color: Colors.blueAccent[100],
                  padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                )),
            Container(
                margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                child: RaisedButton(
                  onPressed: () => getImage(),
                  child: Text('Click Here To Select Video From Gallery'),
                  textColor: Colors.white,
                  color: Colors.blueAccent[100],
                  padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                )),
            Container(
                margin: EdgeInsets.fromLTRB(0, 0, 0, 0),
                child: RaisedButton(
                  onPressed: () => _getbehaviourType(_video!),
                  child: Text('CheckVideo'),
                  textColor: Colors.white,
                  color: Colors.blueAccent[100],
                  padding: EdgeInsets.fromLTRB(12, 12, 12, 12),
                )),
            SizedBox(
              height: 20,
            ),
            filterLoaded
                ? behaviourType != null
                    ? Text(
                        "Predicted Behaviour is: " + behaviourType.toString())
                    : Container()
                : Center(
                    child: CircularProgressIndicator(),
                  ),
          ]),
        ]));
  }
}
