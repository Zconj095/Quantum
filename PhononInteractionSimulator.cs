using UnityEngine;
using System.Collections.Generic;

public class PhononInteractionSimulator : MonoBehaviour
{
    public GameObject phononPrefab; // Prefab for visualizing phonons
    public int phononCount = 20; // Number of phonons to simulate
    public float simulationAreaSize = 5f; // Size of the simulation area
    public float scatteringProbability = 0.1f; // Probability of phonon scattering per update, added declaration here

    private List<GameObject> phonons = new List<GameObject>();

    void Start()
    {
        GeneratePhonons();
    }

    void Update()
    {
        SimulatePhononMovements(); // Ensure this is called within Update to simulate movements
    }

    void GeneratePhonons()
    {
        for (int i = 0; i < phononCount; i++)
        {
            Vector3 position = new Vector3(Random.Range(-simulationAreaSize, simulationAreaSize),
                                            Random.Range(-simulationAreaSize, simulationAreaSize),
                                            Random.Range(-simulationAreaSize, simulationAreaSize));
            GameObject phonon = (GameObject)Instantiate(phononPrefab, position, Quaternion.identity);
            phonons.Add(phonon);
        }
    }

    void SimulatePhononMovements()
    {
        foreach (GameObject phonon in phonons)
        {
            // Simulate random movement
            phonon.transform.Translate(Random.insideUnitSphere * Time.deltaTime);

            // Simulate scattering
            if (Random.value < scatteringProbability) // Now scatteringProbability is recognized
            {
                // Change direction randomly to represent scattering
                phonon.transform.rotation = Random.rotation;
            }

            // Keep phonons within the simulation area
            phonon.transform.position = new Vector3(
                Mathf.Clamp(phonon.transform.position.x, -simulationAreaSize, simulationAreaSize),
                Mathf.Clamp(phonon.transform.position.y, -simulationAreaSize, simulationAreaSize),
                Mathf.Clamp(phonon.transform.position.z, -simulationAreaSize, simulationAreaSize));
        }
    }
}
