import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class ResultPage extends StatelessWidget {
  final String imagePath;
  final Map<String, dynamic> predictionResult;

  const ResultPage({
    required this.imagePath,
    required this.predictionResult,
    Key? key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final subjectId = predictionResult['subject_id'] ?? 'Inconnu';
    final measured = predictionResult['mean_measured_csa']?.toString() ?? '-';
    final normalized = predictionResult['mean_normalized_csa']?.toString() ?? '-';
    final count = predictionResult['count']?.toString() ?? '-';
    final meanArea = predictionResult['mean_area_avg_from_tsv']?.toString();
    final timestamps = predictionResult['timestamps'] as List<dynamic>? ?? [];
    final values = predictionResult['values'] as List<dynamic>? ?? [];
    final base64Image = predictionResult['plot_base64'];
    final fileName = imagePath.split('/').last;

    return Scaffold(
      appBar: AppBar(
        title: Text(
          'Résultats CSA',
          style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                'Fichier MRI : $fileName',
                style: GoogleFonts.poppins(fontSize: 16),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 20),
              if (base64Image != null && base64Image.isNotEmpty)
                Image.memory(
                  base64Decode(base64Image),
                  height: 200,
                  fit: BoxFit.contain,
                )
              else
                const Icon(Icons.image_not_supported, size: 100, color: Colors.grey),
              const SizedBox(height: 20),
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.lightBlue.shade50,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.blueAccent),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text("✅ Sujet : $subjectId", style: GoogleFonts.poppins(fontSize: 16)),
                    const SizedBox(height: 8),
                    Text("CSA mesurée moyenne : $measured mm²", style: GoogleFonts.poppins()),
                    Text("CSA normalisée moyenne : $normalized mm²", style: GoogleFonts.poppins()),
                    Text(" ", style: GoogleFonts.poppins()),
                    if (meanArea != null)
                      Text("CSA moyenne TSV : $meanArea mm²", style: GoogleFonts.poppins()),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              if (timestamps.isNotEmpty && values.isNotEmpty)
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      " Détails par vertèbre :",
                      style: GoogleFonts.poppins(fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 10),
                    ListView.builder(
                      shrinkWrap: true,
                      physics: const NeverScrollableScrollPhysics(),
                      itemCount: timestamps.length,
                      itemBuilder: (context, index) {
                        final time = timestamps[index].toString();
                        final val = values.length > index ? values[index].toString() : '-';
                        return Text(
                          "$time → $val mm²",
                          style: GoogleFonts.poppins(fontSize: 14),
                        );
                      },
                    ),
                  ],
                ),
              const SizedBox(height: 30),
              ElevatedButton.icon(
                onPressed: () {
                  // TODO: Implémenter l’ouverture du fichier MRI si besoin
                  // Exemple: OpenFile.open(imagePath);
                },
                icon: const Icon(Icons.download),
                label: const Text("Ouvrir le fichier MRI"),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blueAccent,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
