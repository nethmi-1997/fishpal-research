// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'requestConsumption.model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

RequestConsumption _$RequestConsumptionFromJson(Map<String, dynamic> json) =>
    RequestConsumption(
      species: json['species'] as int,
      fishCount: json['fishCount'] as int,
      fishStage: json['fishStage'] as int,
    );

Map<String, dynamic> _$RequestConsumptionToJson(RequestConsumption instance) =>
    <String, dynamic>{
      'species': instance.species,
      'fishCount': instance.fishCount,
      'fishStage': instance.fishStage,
    };
