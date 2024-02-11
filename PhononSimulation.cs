using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PhononSimulation : MonoBehaviour
{
    public GameObject acousticPhononPrefab;
    public GameObject opticalPhononPrefab;
    public GameObject defectPrefab; // Represents a material defect or boundary

    private List<GameObject> phonons = new List<GameObject>();
    private GameObject defect;

    void Start()
    {
        // Create a material defect with explicit casting to GameObject
        defect = (GameObject)Instantiate(defectPrefab, new Vector3(0, 0, 0), Quaternion.identity);

        // Generate phonons
        GeneratePhonons(acousticPhononPrefab, 5, new Vector3(-5, 0, 0)); // Acoustic phonons
        GeneratePhonons(opticalPhononPrefab, 5, new Vector3(5, 0, 0)); // Optical phonons
    }


    void Update()
    {
        // Move phonons and simulate interactions
        foreach (var phonon in phonons)
        {
            // Move phonons towards the defect
            phonon.transform.position = Vector3.MoveTowards(phonon.transform.position, defect.transform.position, Time.deltaTime);

            // Check for collision with the defect and scatter
            if (Vector3.Distance(phonon.transform.position, defect.transform.position) < 0.1f)
            {
                // Simulate scattering by randomly changing the phonon's direction
                phonon.transform.position += new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f), Random.Range(-1f, 1f));
            }
        }
    }

    void GeneratePhonons(GameObject phononPrefab, int count, Vector3 startPosition)
    {
        for (int i = 0; i < count; i++)
        {
            var phonon = (GameObject)Instantiate(phononPrefab, startPosition + new Vector3(i, 0, 0), Quaternion.identity);
            phonons.Add(phonon);
        }
    }

}
