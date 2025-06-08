import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl = 'http://172.20.10.4:5000';

  Future<Map<String, dynamic>> sendMRIData({
    required int sex, // 0 = Homme, 1 = Femme
    required double weight,
    required String filePath,
  }) async {
    try {
      var uri = Uri.parse('$baseUrl/process_and_predict');
      var request = http.MultipartRequest('POST', uri);

      // Required form fields
      request.fields['sex'] = sex.toString();
      request.fields['weight'] = weight.toString();

      // Add MRI file
      request.files.add(await http.MultipartFile.fromPath('mri', filePath));

      // Send the request
      var response = await request.send();
      final responseStr = await response.stream.bytesToString();

      if (response.statusCode == 200) {
        final jsonData = json.decode(responseStr);

        return {
          'success': true,
          'data': {
            'subject_id': jsonData['subject_id'],
            'mean_measured_csa': jsonData['mean_measured_csa'],
            'mean_normalized_csa': jsonData['mean_normalized_csa'],
            'count': jsonData['count'],
            'mean_area_avg_from_tsv': jsonData['mean_area_avg_from_tsv'],
            'timestamps': jsonData['timestamps_from_tsv'],
            'values': jsonData['mean_area_values_from_tsv'],
            'plot_base64': jsonData['prediction_plot_png_base64'],
          }
        };
      } else {
        return {
          'success': false,
          'error': 'Erreur serveur (${response.statusCode})',
          'response': responseStr,
        };
      }
    } catch (e) {
      return {
        'success': false,
        'error': 'Exception lors de l’envoi : $e',
      };
    }
  }
}
